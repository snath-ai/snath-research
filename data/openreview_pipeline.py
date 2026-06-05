"""
OpenReview data pipeline for Snath Research.
============================================
Pulls papers + reviews from OpenReview public API and prepares them for
the divergence router experiment.

Stream A input: paper abstract + TL;DR / author summary (the claims)
Stream B input: aggregated reviewer text (the evidence assessment)
Ground truth:   paper decision (Accept/Reject) + mean reviewer score

Supported venues (OpenReview V2 API):
  ICLR 2022 onwards: NeurIPS.cc/2023/Conference, ICLR.cc/2024/Conference, etc.
  All are fully public — no API key required for read access.

Usage:
    from data.openreview_pipeline import OpenReviewPipeline

    pipe = OpenReviewPipeline(venue="ICLR.cc/2024/Conference")
    papers = pipe.fetch(max_papers=500)
    # papers is a list of PaperRecord dicts

    # Prepare for routing:
    for p in papers:
        claims_text  = p["claims_text"]    # → AbstractClaimsEncoder.encode()
        reviews_text = p["reviews_text"]   # → EvidenceEncoder.encode()
        verdict      = p["verdict"]        # attach after routing
"""
from __future__ import annotations

import time
import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class PaperRecord:
    """One paper from OpenReview, prepared for Snath Research routing."""
    paper_id:       str
    venue:          str
    title:          str
    claims_text:    str     # Stream A input
    reviews_text:   str     # Stream B input (aggregated reviewer text)
    decision:       Optional[str]    = None   # Accept / Reject / Withdrawn
    avg_score:      Optional[float]  = None   # mean reviewer numerical score
    num_reviews:    int              = 0


class OpenReviewPipeline:
    """
    Fetches papers and reviews from the OpenReview V2 public API.

    No API key required for public venues (ICLR, NeurIPS, etc.).

    Args:
        venue:       OpenReview venue ID, e.g. "ICLR.cc/2024/Conference"
        sleep_sec:   Polite delay between API calls (default 0.5s).
    """

    BASE_URL = "https://api2.openreview.net"

    def __init__(self, venue: str = "ICLR.cc/2024/Conference", sleep_sec: float = 0.5):
        self.venue     = venue
        self.sleep_sec = sleep_sec

    def _get(self, endpoint: str, params: dict) -> dict:
        """HTTP GET with retry and polite rate limiting."""
        import urllib.request
        import urllib.parse

        url  = f"{self.BASE_URL}{endpoint}?{urllib.parse.urlencode(params)}"
        time.sleep(self.sleep_sec)
        with urllib.request.urlopen(url, timeout=30) as resp:
            return json.loads(resp.read().decode())

    def _fetch_decisions(self) -> Dict[str, dict]:
        """
        Fetch paper decisions (Accept/Reject) and map paper_id → decision info.

        Tries multiple invitation formats across venues:
          - ICLR 2022+:   {venue}/-/Decision
          - NeurIPS 2023+: {venue}/-/Decision  (same)
          - Some venues:   {venue}/-/Acceptance_Decision
          - Fallback:      {venue}/-/Meta_Review

        If all fail, returns {} — infer_winner() falls back to score-only oracle.
        """
        invitations = [
            f"{self.venue}/-/Decision",
            f"{self.venue}/-/Acceptance_Decision",
            f"{self.venue}/-/Meta_Review",
        ]
        for inv in invitations:
            try:
                result = self._get("/notes", {"invitation": inv, "limit": 1000})
                notes  = result.get("notes", [])
                if not notes:
                    continue
                decisions = {}
                for note in notes:
                    forum_id = note.get("forum", "")
                    content  = note.get("content", {})
                    for field in ("decision", "recommendation", "venue_decision"):
                        val = content.get(field, {})
                        if isinstance(val, dict):
                            val = val.get("value", "")
                        if val and isinstance(val, str):
                            decisions[forum_id] = {"decision": val}
                            break
                if decisions:
                    logger.info(f"  Fetched {len(decisions)} decisions via {inv}")
                    return decisions
            except Exception as e:
                logger.debug(f"Decision fetch ({inv}): {e}")

        logger.info("  Decisions unavailable — will use reviewer scores as oracle.")
        return {}

    def _fetch_reviews(self, paper_id: str):
        """
        Fetch all official reviews + decision for a paper.

        Queries ALL forum notes without an invitation filter — ICLR 2024+
        uses per-paper invitations (Submission9504/-/Official_Review) that
        don't match a venue-level invitation string.

        Returns (review_texts, scores, decision).
        """
        review_texts = []
        scores       = []
        decision     = None
        try:
            result = self._get("/notes", {"forum": paper_id, "limit": 30})
            for note in result.get("notes", []):
                content = note.get("content", {})

                # Decision note (has 'decision' key, no long text)
                if decision is None and "decision" in content:
                    d = content.get("decision", {})
                    if isinstance(d, dict):
                        d = d.get("value", "")
                    if d and isinstance(d, str) and len(d) < 200:
                        decision = d

                # Review text — ICLR 2024: 'summary'; older venues: 'review'
                review_text = ""
                for field in ("summary", "review", "main_review",
                              "summary_of_the_paper", "paper_summary"):
                    val = content.get(field, {})
                    if isinstance(val, dict):
                        val = val.get("value", "")
                    if val and isinstance(val, str) and len(val) > 50:
                        review_text = val.strip()
                        break

                # Numerical score — "6: weak accept" → 6.0
                score_val = None
                for field in ("rating", "recommendation", "soundness",
                              "contribution", "overall"):
                    val = content.get(field, {})
                    if isinstance(val, dict):
                        val = val.get("value", "")
                    if val:
                        try:
                            score_val = float(str(val).split(":")[0].strip())
                            break
                        except (ValueError, IndexError):
                            pass

                if review_text and score_val is not None:
                    review_texts.append(review_text)
                    scores.append(score_val)

        except Exception as e:
            logger.debug(f"Reviews fetch error for {paper_id}: {e}")

        return review_texts, scores, decision

    def fetch(self, max_papers: int = 500) -> List[PaperRecord]:
        """
        Fetch papers with abstracts + reviews from the venue.

        Args:
            max_papers: Maximum number of papers to fetch.
        Returns:
            List of PaperRecord, ready for routing.
        """
        logger.info(f"Fetching up to {max_papers} papers from {self.venue}")

        # 1. Fetch paper submissions
        result = self._get("/notes", {
            "invitation": f"{self.venue}/-/Submission",
            "limit":      min(max_papers, 1000),
        })
        submissions = result.get("notes", [])
        logger.info(f"Got {len(submissions)} submissions")

        # 2. Fetch decisions index
        decisions_map = self._fetch_decisions()

        records = []
        for sub in submissions[:max_papers]:
            paper_id = sub.get("id", "")
            content  = sub.get("content", {})

            # Extract abstract (Stream A base)
            abstract = content.get("abstract", {})
            if isinstance(abstract, dict):
                abstract = abstract.get("value", "")
            abstract = abstract.strip() if isinstance(abstract, str) else ""

            # Extract TL;DR / one-sentence summary if available
            tldr = content.get("TL;DR", content.get("one-sentence_summary", {}))
            if isinstance(tldr, dict):
                tldr = tldr.get("value", "")
            tldr = tldr.strip() if isinstance(tldr, str) else ""

            # Claims text = abstract + TL;DR
            claims_text = abstract
            if tldr:
                claims_text = f"{abstract} {tldr}"

            if not claims_text or len(claims_text) < 50:
                continue  # Skip papers with no abstract

            # Fetch reviews + per-paper decision (Stream B)
            review_texts, scores, paper_decision = self._fetch_reviews(paper_id)
            if not review_texts:
                continue  # Skip papers with no reviews

            reviews_text = " [SEP] ".join(review_texts)
            avg_score    = round(sum(scores) / len(scores), 2) if scores else None

            # Decision: prefer bulk fetch; fall back to per-paper note
            d_info   = decisions_map.get(paper_id, {})
            decision = d_info.get("decision", None) or paper_decision

            title = content.get("title", {})
            if isinstance(title, dict):
                title = title.get("value", "")

            records.append(PaperRecord(
                paper_id=paper_id,
                venue=self.venue,
                title=title or "",
                claims_text=claims_text,
                reviews_text=reviews_text,
                decision=decision,
                avg_score=avg_score,
                num_reviews=len(review_texts),
            ))

            if len(records) % 50 == 0:
                logger.info(f"  Fetched {len(records)} papers...")

        logger.info(f"Total records ready: {len(records)}")
        return records

    def save(self, records: List[PaperRecord], path: str) -> None:
        """Save fetched records to JSONL for offline use."""
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(asdict(r)) + "\n")
        logger.info(f"Saved {len(records)} records to {path}")

    @staticmethod
    def load(path: str) -> List[PaperRecord]:
        """Load records from JSONL."""
        records = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    records.append(PaperRecord(**json.loads(line)))
        return records


def infer_winner(decision: Optional[str], avg_score: Optional[float],
                 accept_threshold: float = 6.0,
                 reject_threshold: float = 4.0) -> Optional[str]:
    """
    Infer winner label from OpenReview decision + mean reviewer score.

    Primary oracle: explicit Accept/Reject + score confirmation.
    Fallback oracle: score alone — reviewer scores ARE ground truth
    (ICLR/NeurIPS 1–10 scale: ≤4 strong reject, ≥6 weak accept+).
    Used when the decisions API returns 400 (venue-specific invitation format).

    Returns:
        "claims"  — claims validated (paper accepted with strong reviewer scores)
        "reviews" — reviewers correctly flagged overclaiming (rejected, low scores)
        None      — ambiguous (borderline 4–6, withdrawn, no reviews)
    """
    if avg_score is None:
        return None

    # Primary: explicit decision + score confirmation
    if decision:
        if "Accept" in decision and avg_score >= accept_threshold:
            return "claims"
        if "Reject" in decision and avg_score <= reject_threshold:
            return "reviews"

    # Fallback: score-only oracle (when decision API unavailable)
    if avg_score >= accept_threshold:
        return "claims"
    if avg_score <= reject_threshold:
        return "reviews"

    return None   # borderline (4 < score < 6)
