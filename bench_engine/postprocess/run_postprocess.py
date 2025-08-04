import os
import asyncio
import tempfile
from typing import Optional

from bench_engine.config import (
    SINGLEHOP_JUDGED_JSON_PATH,
    MULTIHOP_JUDGED_JSON_PATH,
    MULTIHOP_TEMP_PATH,
    FINAL_SINGLEHOP_BENCH,
    FINAL_MULTIHOP_BENCH,
)
from format_converter import FormatConverter
from gt_processor import GroundTruthProcessor


class PostProcessor:
    """
    Post-processor for handling annotation and format conversion of single-hop and multi-hop QA data.

    This class provides a complete pipeline for processing both single-hop and multi-hop
    question-answer datasets, including decomposition, ground truth annotation, and merging.
    """

    def __init__(self):
        """
        Initialize the post-processor.

        Sets up the format converter for handling data transformations between
        different QA formats during the processing pipeline.
        """
        self.converter = FormatConverter()

    def create_temp_file(self, suffix: str = ".json") -> str:
        """
        Create temporary file and return its path.

        Args:
            suffix: File suffix for the temporary file

        Returns:
            Path to the created temporary file
        """
        temp_fd, temp_path = tempfile.mkstemp(suffix=suffix)
        os.close(temp_fd)
        return temp_path

    def cleanup_temp_file(self, temp_path: str) -> None:
        """
        Clean up temporary file.

        Args:
            temp_path: Path to the temporary file to be removed
        """
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                print(f"Cleaned up temporary file: {temp_path}")
        except Exception as e:
            print(f"Failed to clean up temporary file: {e}")

    def process_singlehop(self) -> bool:
        """
        Process single-hop QA data with direct ground truth annotation.

        Returns:
            bool: Whether processing was successful
        """
        print("=" * 50)
        print("Starting single-hop QA data processing...")
        print(f"Input file: {SINGLEHOP_JUDGED_JSON_PATH}")
        print(f"Output file: {FINAL_SINGLEHOP_BENCH}")

        try:
            if not os.path.exists(SINGLEHOP_JUDGED_JSON_PATH):
                print(f"Error: Input file does not exist - {SINGLEHOP_JUDGED_JSON_PATH}")
                return False

            output_dir = os.path.dirname(FINAL_SINGLEHOP_BENCH)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created output directory: {output_dir}")

            processor = GroundTruthProcessor(
                input_file=SINGLEHOP_JUDGED_JSON_PATH,
                output_file=FINAL_SINGLEHOP_BENCH
            )

            processor.process()

            if os.path.exists(FINAL_SINGLEHOP_BENCH):
                print(f"Single-hop QA data processing successful! Output: {FINAL_SINGLEHOP_BENCH}")
                return True
            else:
                print("Error: Single-hop QA data processing failed, output file not generated")
                return False

        except Exception as e:
            print(f"Error occurred while processing single-hop QA data: {str(e)}")
            return False

    def process_multihop(self) -> bool:
        """
        Process multi-hop QA data through decomposition, annotation, and merging steps.

        Returns:
            bool: Whether processing was successful
        """
        print("=" * 50)
        print("Starting multi-hop QA data processing...")
        print(f"Input file: {MULTIHOP_JUDGED_JSON_PATH}")
        print(f"Intermediate file: {MULTIHOP_TEMP_PATH}")
        print(f"Output file: {FINAL_MULTIHOP_BENCH}")

        temp_annotated = None

        try:
            if not os.path.exists(MULTIHOP_JUDGED_JSON_PATH):
                print(f"Error: Input file does not exist - {MULTIHOP_JUDGED_JSON_PATH}")
                return False

            # 创建输出目录（如果不存在）
            output_dir = os.path.dirname(FINAL_MULTIHOP_BENCH)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created output directory: {output_dir}")

            # 创建中间文件目录（如果不存在）
            temp_dir = os.path.dirname(MULTIHOP_TEMP_PATH)
            if temp_dir and not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
                print(f"Created intermediate directory: {temp_dir}")

            # 创建一个临时文件用于GT标注结果
            temp_annotated = self.create_temp_file()
            print(f"Created temporary file for annotation: {temp_annotated}")

            # 步骤1: 分解多跳问题为单个问题，使用配置的中间路径
            print("\nStep 1: Decomposing multi-hop questions...")
            if not self.converter.process_file(MULTIHOP_JUDGED_JSON_PATH, MULTIHOP_TEMP_PATH):
                print("Error: Multi-hop question decomposition failed")
                return False
            print(f"Multi-hop question decomposition completed, saved to: {MULTIHOP_TEMP_PATH}")

            # 步骤2: 对分解后的单个问题进行GT标注
            print("\nStep 2: Ground truth annotation of decomposed questions...")
            processor = GroundTruthProcessor(
                input_file=MULTIHOP_TEMP_PATH,
                output_file=temp_annotated
            )
            processor.process()

            if not os.path.exists(temp_annotated):
                print("Error: Ground truth annotation failed, temporary file not generated")
                return False
            print("Ground truth annotation completed")

            # 步骤3: 合并标注后的结果回原始格式
            print("\nStep 3: Merging annotation results...")
            if not self.converter.process_merge_file(
                    temp_annotated,
                    MULTIHOP_JUDGED_JSON_PATH,
                    FINAL_MULTIHOP_BENCH
            ):
                print("Error: Result merging failed")
                return False

            if os.path.exists(FINAL_MULTIHOP_BENCH):
                print(f"Multi-hop QA data processing successful! Output: {FINAL_MULTIHOP_BENCH}")
                return True
            else:
                print("Error: Multi-hop QA data processing failed, final output file not generated")
                return False

        except Exception as e:
            print(f"Error occurred while processing multi-hop QA data: {str(e)}")
            return False

        finally:
            # 只清理临时标注文件，保留中间分解文件
            if temp_annotated:
                self.cleanup_temp_file(temp_annotated)

    def process_all(self) -> bool:
        """
        Process all data including both single-hop and multi-hop QA datasets.

        Returns:
            bool: Whether all processing was successful
        """
        print("Starting post-processing pipeline...")
        print(f"Single-hop data input: {SINGLEHOP_JUDGED_JSON_PATH}")
        print(f"Multi-hop data input: {MULTIHOP_JUDGED_JSON_PATH}")
        print(f"Multi-hop intermediate file: {MULTIHOP_TEMP_PATH}")
        print(f"Single-hop data output: {FINAL_SINGLEHOP_BENCH}")
        print(f"Multi-hop data output: {FINAL_MULTIHOP_BENCH}")

        singlehop_success = False
        multihop_success = False

        try:
            singlehop_success = self.process_singlehop()
        except Exception as e:
            print(f"Single-hop data processing exception: {str(e)}")

        try:
            multihop_success = self.process_multihop()
        except Exception as e:
            print(f"Multi-hop data processing exception: {str(e)}")

        print("=" * 50)
        print("Post-processing pipeline completed")
        print(f"Single-hop data processing: {'Success' if singlehop_success else 'Failed'}")
        print(f"Multi-hop data processing: {'Success' if multihop_success else 'Failed'}")

        overall_success = singlehop_success and multihop_success
        print(f"Overall processing result: {'All successful' if overall_success else 'Partial or complete failure'}")

        return overall_success


def main():
    """
    Main function entry point for running the complete post-processing pipeline.

    Creates a PostProcessor instance and executes the full processing workflow
    for both single-hop and multi-hop QA datasets with error handling.
    """
    processor = PostProcessor()

    try:
        success = processor.process_all()
        exit_code = 0 if success else 1

        if success:
            print("\nAll data processing completed!")
        else:
            print("\nData processing has failed items, please check the error messages above.")

        exit(exit_code)

    except KeyboardInterrupt:
        print("\nUser interrupted the processing pipeline")
        exit(1)
    except Exception as e:
        print(f"\nUnexpected error occurred in processing pipeline: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()