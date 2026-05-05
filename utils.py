import os, re, unicodedata, io
import fitz, cv2, torch, pytesseract, camelot
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from transformers import TableTransformerForObjectDetection, DetrImageProcessor

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r"\(cid:\d+\)", " ", text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"-\s*\n", "", text)
    text = text.replace("\n", " ")
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", text)
    text = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def ocr_image(pil_img, psm=6):
    try:
        txt = pytesseract.image_to_string(pil_img, config=f"--psm {psm}")
    except Exception:
        txt = ""
    return clean_text(txt)

def table_to_markdown(table):
    if not table or len(table) == 0:
        return ""
    header = table[0]
    md = "| " + " | ".join([h if h else "" for h in header]) + " |\n"
    md += "| " + " | ".join(["---"] * len(header)) + " |\n"
    for row in table[1:]:
        md += "| " + " | ".join([c if c else "" for c in row]) + " |\n"
    return md

def table_to_latex(table):
    if not table or len(table) == 0:
        return ""
    ncols = len(table[0])
    latex = "\\begin{tabular}{" + " | ".join(["l"] * ncols) + "}\\n\\hline\\n"
    for row in table:
        latex += " & ".join([cell.replace("\n", " ") if cell else "" for cell in row]) + " \\\\n\\hline\\n"
    latex += "\\end{tabular}"
    return latex

def extract_table_with_camelot(pdf_path, page_no):
    try:
        tables = camelot.read_pdf(pdf_path, pages=str(page_no), flavor="lattice")
        if tables and len(tables) > 0:
            df = tables[0].df
            return df.values.tolist()
    except Exception:
        return None
    return None

def extract_with_yolo_and_table(pdf_path, detector, max_pages=None, dpi=200, table_conf=0.5):
    docs = []
    pdf_doc = fitz.open(pdf_path)
    for pageno, page in tqdm(enumerate(pdf_doc, start=1), total=len(pdf_doc), desc="Extract pages"):
        if max_pages and pageno > max_pages:
            break
        pix = page.get_pixmap(dpi=dpi)
        page_img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        page_cv = cv2.cvtColor(np.array(page_img), cv2.COLOR_RGB2BGR)
        try:
            res = detector.predict(page_cv, imgsz=1024, conf=0.25, device="cpu")[0]
        except Exception:
            res = None
        if res is None:
            docs.append({"page": pageno, "type": "text", "score": 1.0, "content": ocr_image(page_img)})
            continue
        boxes = res.boxes.xyxy.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy().astype(int)
        scores = res.boxes.conf.cpu().numpy()
        for idx, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
            x1, y1, x2, y2 = map(int, box)
            crop = page_img.crop((x1, y1, x2, y2))
            label = detector.names[int(cls)] if hasattr(detector, "names") else str(int(cls))
            if "text" in label.lower():
                docs.append({"page": pageno, "type": "text", "score": float(score), "content": ocr_image(crop)})
            elif "figure" in label.lower():
                fname = f"page{pageno}_figure_{idx}.png"
                crop.save(os.path.join("./pdf_images", fname))
                docs.append({"page": pageno, "type": "figure", "score": float(score), "content": f"[Image: {fname}]"})
            elif "table" in label.lower():
                table = extract_table_with_camelot(pdf_path, pageno)
                if table:
                    docs.append({"page": pageno, "type": "table", "score": float(score), "content": table_to_markdown(table), "content_latex": table_to_latex(table)})
                else:
                    docs.append({"page": pageno, "type": "table", "score": float(score), "content": "[TABLE OCR fallback] " + ocr_image(crop, psm=6)})
            else:
                docs.append({"page": pageno, "type": label, "score": float(score), "content": ocr_image(crop)})
    pdf_doc.close()
    return docs
