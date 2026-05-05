import argparse
import sys
import logging
import sys

from processor import SiteChangeProcessor
def setup_logging():
    """Configures logging to output to both a file and the console."""
    log_format = (
        '%(asctime)s\t'       # Added timestamp
        '%(levelname)s:\t'
        '%(filename)s:'
        '%(funcName)s():'
        '%(lineno)d\t'
        '%(message)s'
    )
    
    logging.basicConfig(
        level=logging.DEBUG,
        format=log_format,
        # datefmt='%Y-%m-%d-%H:%M:%S',  # Specifies the yyyy-mm-dd-h-m-s format
        handlers=[
            logging.FileHandler('runtime.log', mode='w'),
            logging.StreamHandler(sys.stdout) # Logs to console
        ]
    )

def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(
        description="Site Change Detection Heatmap Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--metadata", type=str, default="/home/ubuntu/adrian/drone_AI/progress_monitoring/data/new-c1-r2_2_2026-02-03/metadata.json")
    parser.add_argument("--new_video", type=str, default="/home/ubuntu/adrian/drone_AI/progress_monitoring/data/new-c1-r2_2_2026-02-16.MP4")
    parser.add_argument("--new_srt", type=str, default="/home/ubuntu/adrian/drone_AI/progress_monitoring/data/new-c1-r2_2_2026-02-16.SRT")
    parser.add_argument("--ortho_image", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="comparison_results_2/")
    parser.add_argument("--out_video_name", type=str, default="difference_summary.mp4")
    args = parser.parse_args()

    logging.info("Initializing Site Change Processor...")
    
    try:
        detector = SiteChangeProcessor(
            metadata_path=args.metadata,
            new_video_path=args.new_video,
            new_srt_path=args.new_srt,
            output_dir=args.output_dir,
            ortho_image_path=args.ortho_image,
            out_video_name=args.out_video_name
        )
        
        logging.info("Starting video processing pipeline...")
        detector.process()
        logging.info("Pipeline execution completed successfully.")
        
    except Exception as e:
        logging.error(f"An error occurred during processing: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()