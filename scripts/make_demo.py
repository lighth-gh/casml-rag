"""Create a tiny original test book. This is NOT the CASML competition corpus."""
from pathlib import Path
import csv
import json
import pymupdf as fitz


def create_demo(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    chapters = [
        ("Memory", "Working memory temporarily holds and manipulates information needed for a current task. "
         "For example, a learner may keep two numbers in mind while adding them. Long-term memory can retain "
         "information for much longer periods. A retrieval cue is a hint that helps someone access stored information. "
         "These short paragraphs were written specifically to test source tracing in a software demonstration."),
        ("Learning", "Spaced practice distributes study sessions across multiple occasions instead of combining them "
         "into one session. In this fictional classroom example, Lina reviews her vocabulary on Monday, Wednesday, "
         "and Friday. Retrieval practice means trying to recall information rather than only rereading it. "
         "The demo describes study strategies so that different questions retrieve different pages."),
        ("Research", "A controlled comparison keeps relevant conditions similar while changing the factor being studied. "
         "In a fictional plant experiment, researchers compare two light conditions while keeping water and soil the same. "
         "The independent variable is the factor deliberately changed. The dependent variable is the measured outcome. "
         "This example is invented for testing and is not a passage from the CASML textbook.")]
    doc = fitz.open()
    font = fitz.Font("helv")
    cover = doc.new_page(width=595, height=842)
    cover.insert_font(fontname="DemoFont", fontbuffer=font.buffer)
    cover.insert_text((55, 90), "CASML B0 - Original Demo Book", fontsize=22, fontname="DemoFont")
    cover.insert_textbox(fitz.Rect(55, 140, 540, 350),
        "Synthetic data for a software integration demonstration.\n\n"
        "This file is NOT the competition book.\n"
        "The cover is PDF page 1. Printed content pages start at 1 on PDF page 2.\n\n"
        "Use this small document to verify the pipeline and inspect source references.", fontsize=13, fontname="DemoFont")
    for i, (title, body) in enumerate(chapters, 1):
        page = doc.new_page(width=595, height=842)
        page.insert_font(fontname="DemoFont", fontbuffer=font.buffer)
        page.insert_text((55, 85), title, fontsize=22, fontname="DemoFont")
        page.insert_textbox(fitz.Rect(55, 130, 535, 600), body, fontsize=13, lineheight=1.6, fontname="DemoFont")
        page.insert_text((295, 790), str(i), fontsize=11, fontname="DemoFont")
    doc.set_toc([[1, "Demo chapters", 2], [2, "Memory", 2], [2, "Learning", 3], [2, "Research", 4]])
    doc.set_page_labels([{"startpage": 0, "prefix": "Cover", "style": "", "firstpagenum": 1},
                         {"startpage": 1, "style": "D", "firstpagenum": 1}])
    doc.save(destination / "demo_book.pdf")
    doc.close()
    queries = [{"query_id": "Q001", "question": "What does working memory do?"},
               {"query_id": "Q002", "question": "What is spaced practice?"},
               {"query_id": "Q003", "question": "What is the independent variable in an experiment?"}]
    (destination / "queries.json").write_text(json.dumps(queries, indent=2), encoding="utf-8")
    with open(destination / "sample_submission.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ID", "context", "answer", "references"])
        for q in queries:
            w.writerow([q["query_id"], "", "", ""])
    with open(destination / "page_map_override.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pdf_page", "printed_page", "section_path"])
        w.writerow([1, "Cover", ""])
        for i, (title, _) in enumerate(chapters, 2):
            w.writerow([i, str(i - 1), f"Demo chapters/{title}"])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="examples")
    create_demo(parser.parse_args().out)
