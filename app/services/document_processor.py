"""Document processing service for extracting text from PDF and TXT files."""

import logging
from pathlib import Path
from typing import Optional

import pdfplumber
import PyPDF2

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Service for extracting text from documents."""

    @staticmethod
    def extract_text_from_pdf(file_path: str, use_pdfplumber: bool = True) -> str:
        """
        Extract text from PDF file.

        Args:
            file_path: Path to PDF file
            use_pdfplumber: If True, use pdfplumber (better quality), else PyPDF2

        Returns:
            str: Extracted text
        """
        try:
            if use_pdfplumber:
                return DocumentProcessor._extract_with_pdfplumber(file_path)
            else:
                return DocumentProcessor._extract_with_pypdf2(file_path)
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            # Try fallback method
            try:
                logger.info("Trying fallback PDF extraction method...")
                if use_pdfplumber:
                    return DocumentProcessor._extract_with_pypdf2(file_path)
                else:
                    return DocumentProcessor._extract_with_pdfplumber(file_path)
            except Exception as fallback_error:
                logger.error(f"Fallback extraction also failed: {fallback_error}")
                raise Exception(f"Failed to extract text from PDF: {e}")

    @staticmethod
    def _extract_with_pdfplumber(file_path: str) -> str:
        """Extract text using pdfplumber (better for complex PDFs)."""
        text_parts = []

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
                    else:
                        logger.warning(f"No text extracted from page {page_num}")
                except Exception as e:
                    logger.warning(f"Error extracting page {page_num}: {e}")
                    continue

        full_text = "\n\n".join(text_parts)
        logger.info(f"Extracted {len(full_text)} characters from PDF using pdfplumber")
        return full_text

    @staticmethod
    def _extract_with_pypdf2(file_path: str) -> str:
        """Extract text using PyPDF2 (faster but less accurate)."""
        text_parts = []

        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)

            for page_num in range(total_pages):
                try:
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
                    else:
                        logger.warning(f"No text extracted from page {page_num + 1}")
                except Exception as e:
                    logger.warning(f"Error extracting page {page_num + 1}: {e}")
                    continue

        full_text = "\n\n".join(text_parts)
        logger.info(f"Extracted {len(full_text)} characters from PDF using PyPDF2")
        return full_text

    @staticmethod
    def extract_text_from_txt(file_path: str, encoding: str = "utf-8") -> str:
        """
        Extract text from TXT file.

        Args:
            file_path: Path to TXT file
            encoding: File encoding (default: utf-8)

        Returns:
            str: File content

        """
        try:
            with open(file_path, "r", encoding=encoding) as file:
                text = file.read()

            logger.info(f"Extracted {len(text)} characters from TXT file")
            return text

        except UnicodeDecodeError:
            # Try with different encoding
            logger.warning(f"UTF-8 decoding failed, trying latin-1")
            try:
                with open(file_path, "r", encoding="latin-1") as file:
                    text = file.read()
                return text
            except Exception as e:
                logger.error(f"Error reading TXT file with fallback encoding: {e}")
                raise
        except Exception as e:
            logger.error(f"Error extracting text from TXT: {e}")
            raise

    @staticmethod
    def extract_text(file_path: str, file_extension: str) -> str:
        """
        Extract text from file based on extension.

        Args:
            file_path: Path to file
            file_extension: File extension (.pdf, .txt)

        Returns:
            str: Extracted text

        """
        extension = file_extension.lower()

        if extension == ".pdf":
            return DocumentProcessor.extract_text_from_pdf(file_path)
        if extension == ".txt":
            return DocumentProcessor.extract_text_from_txt(file_path)
        raise ValueError(f"Unsupported file extension: {extension}")

    @staticmethod
    def validate_file(file_path: str, max_size: int) -> tuple[bool, Optional[str]]:
        """
        Validate file exists and size is within limits.

        Args:
            file_path: Path to file
            max_size: Maximum file size in bytes

        Returns:
            tuple: (is_valid, error_message)

        """
        path = Path(file_path)

        if not path.exists():
            return False, "File does not exist"

        if not path.is_file():
            return False, "Path is not a file"

        file_size = path.stat().st_size
        if file_size > max_size:
            return (
                False,
                f"File size ({file_size} bytes) exceeds maximum ({max_size} bytes)",
            )

        if file_size == 0:
            return False, "File is empty"

        return True, None


# Global document processor instance
document_processor = DocumentProcessor()


def get_document_processor() -> DocumentProcessor:
    """
    Dependency function to get document processor.

    Returns:
        DocumentProcessor: Document processor instance

    """
    return document_processor
