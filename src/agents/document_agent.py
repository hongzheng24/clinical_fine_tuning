import re
import pdfplumber
from src.utils.state import PipelineState


def _clean(text: str) -> str:
    '''
    Clean raw text. Collapse runs of blank lines, join wrapped lines within a paragraph,
    and clean up double spaces.

    Parameters
    ==========
    text: str
        Raw text to be cleaned.
    
    Returns
    =======
    processed_text: str
        Processed text.
    '''
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)              # collapse runs of blank lines
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)         # join wrapped lines within a paragraph
    text = re.sub(r"[ \t]+", " ", text)                  # clean up any double spaces just introduced
    return text.strip()


def _scan_ocr_pdf(path: str) -> str:
    # TODO: Implement OCR scanning for PDF documents
    '''
    Scan PDF file and convert to raw text.
    '''
    pass

def load_document(state: PipelineState) -> dict:
    '''
    Load document from state. Run OCR if PDF file, else load raw text.

    Parameter
    =========
    state: PipelineState
        State to load document from.

    Returns
    =======
    document: dict
        Map containing cleaned text.
    '''
    if state.get('is_pdf'):
        pass
    else:
        text = state['raw_input']
        trace = "loaded raw text input (no OCR needed)"

        return {'text': _clean(text), 'trace': [f'[DocumentAgent] {trace}']}
    