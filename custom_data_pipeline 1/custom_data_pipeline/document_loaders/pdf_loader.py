import fitz
import os
from llama_index.core import Document
from document_loaders.base_loader import SimpleDocumentLoader
from core.enums import DocumentType, TableParsingStrategy
from utils import process_text_blocks, extract_text_around_item, process_graph, is_graph

class PDFDocumentLoader(SimpleDocumentLoader):
    """Loader for PDF documents."""
    def _load_data(self, pdf_file) -> list[Document]:
        all_pdf_documents = []
        ongoing_tables = {}

        try:
            f = fitz.open(stream=pdf_file.read(), filetype="pdf")
        except Exception as e:
            print(f"Error opening or processing the PDF file: {e}")
            return []

        for i in range(len(f)):
            page = f[i]
            text_blocks = [block for block in page.get_text("blocks", sort=True)
                           if block[-1] == 0 and not (block[1] < page.rect.height * 0.1 or block[3] > page.rect.height * 0.9)]
            grouped_text_blocks = process_text_blocks(text_blocks)

            table_docs, table_bboxes, ongoing_tables = self._parse_all_tables(pdf_file.name, page, i, text_blocks, ongoing_tables)
            all_pdf_documents.extend(table_docs)

            image_docs = self._parse_all_images(pdf_file.name, page, i, text_blocks)
            all_pdf_documents.extend(image_docs)

            for text_block_ctr, (heading_block, content) in enumerate(grouped_text_blocks, 1):
                heading_bbox = fitz.Rect(heading_block[:4])
                if not any(heading_bbox.intersects(table_bbox) for table_bbox in table_bboxes):
                    bbox = {"x1": heading_block[0], "y1": heading_block[1], "x2": heading_block[2], "x3": heading_bbox[3]}
                    text_doc = Document(
                        text=f"{heading_block[4]}\n{content}",
                        metadata={
                            **bbox,
                            "type": DocumentType.TEXT.value,
                            "page_num": i,
                            "source": f"{pdf_file.name[:-4]}-page{i}-block{text_block_ctr}"
                        },
                        id_=f"{pdf_file.name[:-4]}-page{i}-block{text_block_ctr}"
                    )
                    all_pdf_documents.append(text_doc)

        f.close()
        return all_pdf_documents

    def _parse_all_tables(self, filename, page, pagenum, text_blocks, ongoing_tables):
        """Extract tables from a PDF page."""
        table_docs = []
        table_bboxes = []
        try:
            tables = page.find_tables(horizontal_strategy=TableParsingStrategy.LINES_STRICT_HORIZONTAL.value,
                                       vertical_strategy=TableParsingStrategy.LINES_STRICT_VERTICAL.value)
            for tab in tables:
                if not tab.header.external:
                    pandas_df = tab.to_pandas()
                    tablerefdir = os.path.join(os.getcwd(), "vectorstore/table_references")
                    os.makedirs(tablerefdir, exist_ok=True)
                    df_xlsx_path = os.path.join(tablerefdir, f"table{len(table_docs)+1}-page{pagenum}.xlsx")
                    pandas_df.to_excel(df_xlsx_path)
                    bbox = fitz.Rect(tab.bbox)
                    table_bboxes.append(bbox)

                    before_text, after_text = extract_text_around_item(text_blocks, bbox, page.rect.height)

                    table_img = page.get_pixmap(clip=bbox)
                    table_img_path = os.path.join(tablerefdir, f"table{len(table_docs)+1}-page{pagenum}.jpg")
                    table_img.save(table_img_path)
                    description = process_graph(table_img.tobytes())

                    caption = before_text.replace("\n", " ") + description + after_text.replace("\n", " ")
                    if before_text == "" and after_text == "":
                        caption = " ".join(tab.header.names)
                    table_metadata = {
                        "source": f"{filename[:-4]}-page{pagenum}-table{len(table_docs)+1}",
                        "dataframe": df_xlsx_path,
                        "image": table_img_path,
                        "caption": caption,
                        "type": DocumentType.TABLE.value,
                        "page_num": pagenum
                    }
                    all_cols = ", ".join(list(pandas_df.columns.values))
                    doc = Document(text=f"This is a table with the caption: {caption}\nThe columns are {all_cols}", metadata=table_metadata)
                    table_docs.append(doc)
        except Exception as e:
            print(f"Error during table extraction: {e}")
        return table_docs, table_bboxes, ongoing_tables

    def _parse_all_images(self, filename, page, pagenum, text_blocks):
        """Extract images from a PDF page."""
        image_docs = []
        image_info_list = page.get_image_info(xrefs=True)
        page_rect = page.rect

        for image_info in image_info_list:
            xref = image_info['xref']
            if xref == 0:
                continue

            img_bbox = fitz.Rect(image_info['bbox'])
            if img_bbox.width < page_rect.width / 20 or img_bbox.height < page_rect.height / 20:
                continue

            extracted_image = page.parent.extract_image(xref)
            image_data = extracted_image["image"]
            imgrefpath = os.path.join(os.getcwd(), "vectorstore/image_references")
            os.makedirs(imgrefpath, exist_ok=True)
            image_path = os.path.join(imgrefpath, f"image{xref}-page{pagenum}.png")
            with open(image_path, "wb") as img_file:
                img_file.write(image_data)

            before_text, after_text = extract_text_around_item(text_blocks, img_bbox, page.rect.height)
            if before_text == "" and after_text == "":
                continue

            image_description = " "
            if is_graph(image_data):
                image_description = process_graph(image_data)

            caption = before_text.replace("\n", " ") + image_description + after_text.replace("\n", " ")

            image_metadata = {
                "source": f"{filename[:-4]}-page{pagenum}-image{xref}",
                "image": image_path,
                "caption": caption,
                "type": DocumentType.IMAGE.value,
                "page_num": pagenum
            }
            image_docs.append(Document(text="This is an image with the caption: " + caption, metadata=image_metadata))
        return image_docs