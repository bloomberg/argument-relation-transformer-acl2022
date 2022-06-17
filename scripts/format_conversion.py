"""Script to convert Essays/AbstRCT/ECHR/CDCP to unified format"""
import argparse

from dataset import (
    UKPDocument,
    CDCPDocument,
    ECHRDocument,
    AbstCRTDocument
)

dataset_map = {
    'cdcp' : CDCPDocument,
    'essays' : UKPDocument,
    'abst_rct' : AbstCRTDocument,
    'echr' : ECHRDocument
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, 
                        choices=['abst_rct', 'essays', 'echr', 'cdcp'])
    args = parser.parse_args()

    dataset_class = dataset_map[args.dataset]
    dataset_class.make_all_data()

if __name__=='__main__':
    main()

