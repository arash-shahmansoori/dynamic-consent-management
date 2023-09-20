import os


from torch.hub import download_url_to_file
from torch.utils.data import Dataset
from torchaudio.datasets.utils import _load_waveform, extract_archive


SAMPLE_RATE = 16000
_ARCHIVE_CONFIGS = {
    "dev": {
        "archive_name": "vox2_dev_aac.zip",
        "urls": [
            "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partaa",
            # "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partab",
            # "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partac",
            # "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partad",
            # "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partae",
            # "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partaf",
            # "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partag",
            # "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partah",
        ],
        "checksums": [
            "da070494c573e5c0564b1d11c3b20577",
            # "17fe6dab2b32b48abaf1676429cdd06f",
            # "1de58e086c5edf63625af1cb6d831528",
            # "5a043eb03e15c5a918ee6a52aad477f9",
            # "cea401b624983e2d0b2a87fb5d59aa60",
            # "fc886d9ba90ab88e7880ee98effd6ae9",
            # "d160ecc3f6ee3eed54d55349531cb42e",
            # "6b84a81b9af72a9d9eecbb3b1f602e65",
        ],
    },
    "test": {
        "archive_name": "vox2_test_aac.zip",
        "url": "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_test_aac.zip",
        "checksum": "e4d9200107a7bc60f0b620d5dc04c3aab66681b649f9c218380ac43c6c722079",
    },
}


def download_extract_aac(root):
    for archive in ["dev"]:
        # archive_name = _ARCHIVE_CONFIGS[archive]["archive_name"]
        # archive_path = os.path.join(root, archive_name)
        # The zip file of dev data is splited to 4 chunks.
        # Download and combine them into one file before extraction.
        # if archive == "dev":
        #     urls = _ARCHIVE_CONFIGS[archive]["urls"]
        #     checksums = _ARCHIVE_CONFIGS[archive]["checksums"]
        #     with open(archive_path, "wb") as f:
        #         for url, checksum in zip(
        #             urls,
        #             checksums,
        #             # hash_prefix=checksum,
        #         ):
        #             file_path = os.path.join(root, os.path.basename(url))
        #             download_url_to_file(url, file_path)
        #             with open(file_path, "rb") as f_split:
        #                 f.write(f_split.read())
        # else:
        #     url = _ARCHIVE_CONFIGS[archive]["url"]
        #     checksum = _ARCHIVE_CONFIGS[archive]["checksum"]
        #     download_url_to_file(
        #         url,
        #         archive_path,
        #         # hash_prefix=checksum,
        #     )
        # print(archive_path)
        # extract_archive(archive_path)

        print("testing")
