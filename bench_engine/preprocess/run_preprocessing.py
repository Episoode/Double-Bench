import os
import asyncio
from tqdm import tqdm
from document_parser import DocumentParser
from visual_filter import VisualFliter
from bench_engine.config import SOURCE_DIR, TARGET_DIR


async def run_document_parsing():
    """
    Run the document parsing workflow to process all documents in the source directory.

    This function creates a DocumentParser instance and processes all documents
    from SOURCE_DIR to TARGET_DIR with concurrent processing using semaphore control.
    """
    parser = DocumentParser()
    source_dir = SOURCE_DIR
    tgt_dir = TARGET_DIR
    docs = os.listdir(source_dir)

    # Create progress bar and semaphore for concurrency control
    pbar = tqdm(total=len(docs), desc="Processing documents")
    semaphore = asyncio.Semaphore(5)

    async def process_doc_with_progress(doc):
        """
        Process a single document with progress tracking.

        Args:
            doc (str): Document folder name to be processed
        """
        async with semaphore:
            doc_path = os.path.join(source_dir, doc)
            tgt_path = os.path.join(tgt_dir, doc)
            await parser.process_doc(doc_path, tgt_path)
            pbar.update(1)

    # Process all documents concurrently
    await asyncio.gather(*[process_doc_with_progress(doc) for doc in docs])
    pbar.close()


async def run_visual_flitering():
    """
    Run the image filtering workflow to filter images in all processed documents.

    This function creates a VisualFliter instance and processes all documents
    in the TARGET_DIR to filter out unwanted images using AI-based analysis.
    """
    filter_instance = VisualFliter()

    tgt_dir = TARGET_DIR
    docs = os.listdir(tgt_dir)

    # Create progress bar for tracking filtering progress
    pbar = tqdm(total=len(docs), desc="Filtering images")

    async def process_doc_with_progress(doc):
        """
        Process a single document for image filtering with progress tracking.

        Args:
            doc (str): Document folder name to be processed for image filtering
        """
        doc_path = os.path.join(tgt_dir, doc)
        semaphore = asyncio.Semaphore(16)
        await filter_instance.process_dir(doc_path, semaphore)
        pbar.update(1)

    # Process all documents concurrently for image filtering
    await asyncio.gather(*[process_doc_with_progress(doc) for doc in docs])
    pbar.close()


def main():
    """
    Main function that controls the entire preprocessing workflow.

    This function orchestrates the complete document processing pipeline:
    1. Document parsing: converts documents to structured format
    2. Image filtering: filters out unwanted images using AI analysis

    The function runs both steps sequentially to ensure proper data flow.
    """
    print("Starting document parsing...")
    asyncio.run(run_document_parsing())

    print("Starting image filtering...")
    asyncio.run(run_visual_flitering())

    print("Preprocessing completed!")


if __name__ == "__main__":
    main()