from utils import extract_text_around_item, process_text_blocks

# These functions remain largely the same, ensure they are in the utils/__init__.py
# for them to be importable directly from 'utils' if other parts of the system rely on that.

__all__ = ['extract_text_around_item', 'process_text_blocks']