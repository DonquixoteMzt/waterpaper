#!/usr/bin/env python3
"""Programmatically verify paper existence using Semantic Scholar and arXiv APIs."""

import urllib.request
import urllib.parse
import json
import time
import xml.etree.ElementTree as ET

# Candidate papers to verify
papers = [
    {
        "id": 1,
        "title": "Visualizing the Loss Landscape of Neural Nets",
        "authors": "Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer, Tom Goldstein",
        "venue": "NeurIPS 2018",
        "arxiv_id": "1712.09913",
    },
    {
        "id": 2,
        "title": "Unveiling the Basin-Like Loss Landscape in Large Language Models",
        "authors": "multiple authors",
        "venue": "arXiv preprint 2025",
        "arxiv_id": "2505.17646",
    },
    {
        "id": 3,
        "title": "Visualizing high-dimensional loss landscapes with Hessian directions",
        "authors": "Lucas Böttcher, Gregory Wheeler",
        "venue": "Journal of Statistical Mechanics 2024",
        "arxiv_id": "2208.13219",
    },
    {
        "id": 4,
        "title": "Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs",
        "authors": "Timur Garipov, Pavel Izmailov, Dmitrii Podoprikhin, Dmitry P. Vetrov, Andrew G. Wilson",
        "venue": "NeurIPS 2018",
        "arxiv_id": "1802.10026",
    },
    {
        "id": 5,
        "title": "Visualizing, Rethinking, and Mining the Loss Landscape of Deep Neural Networks",
        "authors": "Xin-Chun Li et al.",
        "venue": "arXiv preprint 2024",
        "arxiv_id": "2405.12493",
    },
    {
        "id": 6,
        "title": "Dissecting Hessian: Understanding Common Structure of Hessian in Neural Networks",
        "authors": "Yikai Wu, Xingyu Zhu, Chenwei Wu, Annie Wang, Rong Ge",
        "venue": "arXiv preprint 2020",
        "arxiv_id": "2010.04261",
    },
    {
        "id": 7,
        "title": "Training Trajectories of Language Models Across Scales",
        "authors": "Mengzhou Xia, Mikel Artetxe, Chunting Zhou, Xi Victoria Lin, Ramakanth Pasunuru, Danqi Chen, Luke Zettlemoyer, Veselin Stoyanov",
        "venue": "ACL 2023",
        "arxiv_id": "2212.09803",
    },
    {
        "id": 8,
        "title": "Sharpness-Aware Minimization for Efficiently Improving Generalization",
        "authors": "Pierre Foret, Ariel Kleiner, Hossein Mobahi, Behnam Neyshabur",
        "venue": "ICLR 2021",
        "arxiv_id": "2010.01412",
    },
    {
        "id": 9,
        "title": "Same Pre-training Loss, Better Downstream: Implicit Bias Matters for Language Models",
        "authors": "Hong Liu, Sang Michael Xie, Zhiyuan Li, Tengyu Ma",
        "venue": "ICML 2023",
        "arxiv_id": "2210.14199",
    },
    {
        "id": 10,
        "title": "A Scalable Measure of Loss Landscape Curvature for Analyzing the Training Dynamics of LLMs",
        "authors": "Dayal Singh Kalra et al.",
        "venue": "arXiv preprint 2026",
        "arxiv_id": "2601.16979",
    },
    {
        "id": 11,
        "title": "Qualitatively characterizing neural network optimization problems",
        "authors": "Ian J. Goodfellow, Oriol Vinyals, Andrew M. Saxe",
        "venue": "ICLR 2015",
        "arxiv_id": "1412.6544",
    },
    {
        "id": 12,
        "title": "Loss Landscape Degeneracy Drives Stagewise Development in Transformers",
        "authors": "multiple authors",
        "venue": "arXiv preprint / ICML 2024",
        "arxiv_id": "2402.02364",
    },
    {
        "id": 13,
        "title": "An Investigation into Neural Net Optimization via Hessian Eigenvalue Density",
        "authors": "Behrooz Ghorbani, Shankar Krishnan, Ying Xiao",
        "venue": "ICML 2019",
        "arxiv_id": "1901.10159",
    },
]


def verify_via_arxiv(arxiv_id):
    """Verify paper existence via arXiv API."""
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as response:
            data = response.read().decode("utf-8")
        # Parse XML
        root = ET.fromstring(data)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)
        if entries:
            entry = entries[0]
            title = entry.find("atom:title", ns)
            authors = entry.findall("atom:author", ns)
            if title is not None:
                title_text = title.text.strip().replace("\n", " ")
                author_names = []
                for a in authors:
                    name = a.find("atom:name", ns)
                    if name is not None:
                        author_names.append(name.text.strip())
                return {
                    "found": True,
                    "source": "arXiv",
                    "title": title_text,
                    "authors": author_names,
                }
        return {"found": False, "source": "arXiv", "title": None, "authors": []}
    except Exception as e:
        return {"found": False, "source": "arXiv", "error": str(e)}


def verify_via_semantic_scholar(title):
    """Verify paper existence via Semantic Scholar API."""
    encoded = urllib.parse.quote(title)
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded}&limit=3&fields=title,authors,venue,year,citationCount"
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "PaperVerifier/1.0")
        with urllib.request.urlopen(req, timeout=15) as response:
            data = json.loads(response.read().decode("utf-8"))
        if data.get("data"):
            for paper in data["data"]:
                # Check title similarity
                p_title = paper.get("title", "").lower().strip()
                q_title = title.lower().strip()
                if q_title in p_title or p_title in q_title or _similar(p_title, q_title):
                    author_names = [
                        a.get("name", "") for a in paper.get("authors", [])
                    ]
                    return {
                        "found": True,
                        "source": "Semantic Scholar",
                        "title": paper.get("title"),
                        "authors": author_names,
                        "venue": paper.get("venue", ""),
                        "year": paper.get("year"),
                        "citations": paper.get("citationCount", 0),
                    }
        return {"found": False, "source": "Semantic Scholar"}
    except Exception as e:
        return {"found": False, "source": "Semantic Scholar", "error": str(e)}


def _similar(a, b):
    """Simple word-overlap similarity check."""
    words_a = set(a.split())
    words_b = set(b.split())
    if not words_a or not words_b:
        return False
    overlap = len(words_a & words_b) / max(len(words_a), len(words_b))
    return overlap > 0.7


results = []
for paper in papers:
    print(f"\n{'='*80}")
    print(f"Verifying Paper #{paper['id']}: {paper['title']}")
    print(f"Expected venue: {paper['venue']}")
    print(f"arXiv ID: {paper['arxiv_id']}")

    # Verify via arXiv
    arxiv_result = verify_via_arxiv(paper["arxiv_id"])
    time.sleep(1)  # rate limit

    # Verify via Semantic Scholar
    ss_result = verify_via_semantic_scholar(paper["title"])
    time.sleep(1)  # rate limit

    # Determine status
    arxiv_confirmed = arxiv_result.get("found", False)
    ss_confirmed = ss_result.get("found", False)

    if arxiv_confirmed:
        status = "CONFIRMED"
        source = "arXiv"
        verified_title = arxiv_result.get("title", "")
        verified_authors = arxiv_result.get("authors", [])
    elif ss_confirmed:
        status = "CONFIRMED"
        source = "Semantic Scholar"
        verified_title = ss_result.get("title", "")
        verified_authors = ss_result.get("authors", [])
    else:
        status = "UNCONFIRMED"
        source = "None"
        verified_title = ""
        verified_authors = []

    citations = ss_result.get("citations", "N/A") if ss_confirmed else "N/A"

    result = {
        "id": paper["id"],
        "title": paper["title"],
        "expected_venue": paper["venue"],
        "arxiv_id": paper["arxiv_id"],
        "status": status,
        "verification_source": source,
        "verified_title": verified_title,
        "verified_authors": verified_authors,
        "citations": citations,
    }
    results.append(result)

    print(f"  arXiv: {'FOUND' if arxiv_confirmed else 'NOT FOUND'}")
    if arxiv_confirmed:
        print(f"    Title: {arxiv_result.get('title', '')}")
        print(f"    Authors: {', '.join(arxiv_result.get('authors', []))}")
    print(f"  Semantic Scholar: {'FOUND' if ss_confirmed else 'NOT FOUND'}")
    if ss_confirmed:
        print(f"    Title: {ss_result.get('title', '')}")
        print(f"    Citations: {ss_result.get('citations', 'N/A')}")
    print(f"  STATUS: {status} (via {source})")

# Summary
print(f"\n\n{'='*80}")
print("VERIFICATION SUMMARY")
print(f"{'='*80}")
confirmed = [r for r in results if r["status"] == "CONFIRMED"]
discarded = [r for r in results if r["status"] != "CONFIRMED"]
print(f"Total papers: {len(results)}")
print(f"Confirmed: {len(confirmed)}")
print(f"Discarded: {len(discarded)}")
print()
for r in results:
    print(f"  [{r['status']}] #{r['id']}: {r['title']}")
    print(f"           Venue: {r['expected_venue']} | Citations: {r['citations']}")

# Save results
with open("verification_results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("\nResults saved to verification_results.json")
