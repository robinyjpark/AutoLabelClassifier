from setuptools import setup, find_packages
import sys
from os import path

short_description = (
    "Automatic Radiological Report Labelling Using Large Language Models"
)

this_directory = path.abspath(path.dirname(__file__))
kwargs = {"encoding": "utf-8"} if sys.version_info.major == 3 else {}
with open(path.join(this_directory, "README.md"), **kwargs) as f:
    __long_description__ = f.read()
__long_description_content_type__ = "text/markdown"

setup(
    name="AutoLabelClassifier",
    author="Robin Park, Rhydian Windsor",
    author_email="robin@robots.ox.ac.uk, rhydian@robots.ox.ac.uk",
    description=short_description,
    long_description=__long_description__,
    long_description_content_type=__long_description_content_type__,
    version="0.1",
    python_requires=">=3.8",
    packages=find_packages(),
    entry_points={
        "console_scripts": "auto_label_classifier = AutoLabelClassifier.main:main"
    },
)
