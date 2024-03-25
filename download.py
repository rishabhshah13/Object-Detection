import gdown
from zipfile import ZipFile
import os
import argparse

def main(args):
    # Download raw data
    if args.download_data and not os.path.exists('ObjectDetectionDataset.zip'):
        output = "ObjectDetectionDataset.zip"
        id = "1Ljjye8lyZHs6vzSLlx-fDlxbix8Nf3-w"
        gdown.download(id=id, output=output)

        with ZipFile('ObjectDetectionDataset.zip', 'r') as zipObj:
            zipObj.extractall('')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download data and models for scene recognition.')
    parser.add_argument('--download_data', type=bool, default=False, help='Download raw data.')
    parser.add_argument('--download_models', type=bool, default=False, help='Download best models.')
    args = parser.parse_args()
    main(args)