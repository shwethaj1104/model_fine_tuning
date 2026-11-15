"""Package final-test code and a separate frozen adapter restore ZIP."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from scripts.prepare_colab import prepare, ROOT
from evaluation.final_test import verify


def main():
    bundle=ROOT/'artifacts/20261115T073319089436Z-lora-results'
    reference,adapter,_=verify(bundle,ROOT/'training/final_test_reference.json')
    project=prepare(ROOT/'artifacts/colab-final-test-project.zip')
    restore=ROOT/'artifacts/final-test-adapter.zip'
    with ZipFile(restore,'w',ZIP_DEFLATED) as archive:
        for name in reference['adapter_files_sha256']:
            archive.write(adapter/name,'training/adapter/'+name)
    print(project)
    print(restore)


if __name__=='__main__':main()
