import fitz
from core.base import TextProcessor
from core.enums import TextBlockType

class FitzTextProcessor(TextProcessor):
    """Text processor using the Fitz library."""

    def extract_text_around_item(self, text_blocks: list, bbox: fitz.Rect, page_height: float, threshold_percentage: float = 0.1) -> tuple[str, str]:
        """Extract text above and below a given bounding box on a page."""
        before_text, after_text = "", ""
        vertical_threshold_distance = page_height * threshold_percentage
        horizontal_threshold_distance = bbox.width * threshold_percentage

        for block in text_blocks:
            block_bbox = fitz.Rect(block[:4])
            vertical_distance = min(abs(block_bbox.y1 - bbox.y0), abs(block_bbox.y0 - bbox.y1))
            horizontal_overlap = max(0, min(block_bbox.x1, bbox.x1) - max(block_bbox.x0, bbox.x0))

            if vertical_distance <= vertical_threshold_distance and horizontal_overlap >= -horizontal_threshold_distance:
                if block_bbox.y1 < bbox.y0 and not before_text:
                    before_text = block[4]
                elif block_bbox.y0 > bbox.y1 and not after_text:
                    after_text = block[4]
                    break

        return before_text, after_text

    def process_text(self, text_blocks: list, char_count_threshold: int = 500) -> list[tuple[tuple, str]]:
        """Group text blocks based on a character count threshold."""
        current_group = []
        grouped_blocks = []
        current_char_count = 0

        for block in text_blocks:
            if block[-1] == TextBlockType.TEXT.value:  # Check if the block is of text type
                block_text = block[4]
                block_char_count = len(block_text)

                if current_char_count + block_char_count <= char_count_threshold:
                    current_group.append(block)
                    current_char_count += block_char_count
                else:
                    if current_group:
                        grouped_content = "\n".join([b[4] for b in current_group])
                        grouped_blocks.append((current_group[0], grouped_content))
                    current_group = [block]
                    current_char_count = block_char_count

        # Append the last group
        if current_group:
            grouped_content = "\n".join([b[4] for b in current_group])
            grouped_blocks.append((current_group[0], grouped_content))

        return grouped_blocks