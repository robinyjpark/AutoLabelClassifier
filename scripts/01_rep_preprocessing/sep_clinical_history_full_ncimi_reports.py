import logging
import string
import re
import bioc
import pandas as pd
import hydra 

from pathlib2 import Path
from omegaconf import DictConfig


def printable(s, func=None):
    """
    Return string of ASCII string which is considered printable.

    Args:
        s(str): string
        func: function to convert non-ASCII characters
    """
    out = ''
    for c in s:
        if c in string.printable:
            out += c
        elif func is not None:
            out += func(c)
        else:
            logging.warning('Cannot convert char: %s', c)
    return out


def text2document(id, text):
    """
    Convert text to a BioCDocument instance

    Args:
        id (str): BioCDocument id
        text (str): text

    Returns:
        BioCDocument: a BioCDocument instance
    """
    document = bioc.BioCDocument()
    document.id = id
    text = printable(text).replace('\r\n', '\n')

    passage = bioc.BioCPassage()
    passage.offset = 0
    passage.text = text
    document.add_passage(passage)

    return document


def text2collection(*sources):
    """
    Returns a BioCCollection containing documents specified in sources.

    Args:
        sources: a list of pathname
    """

    collection = bioc.BioCCollection()
    for pathname in iter(*sources):
        logging.debug('Process %s', pathname)
        try:
            with open(pathname) as fp:
                text = fp.read()
            id = Path(pathname).stem
            document = text2document(id, text)
            collection.add_document(document)
        except:
            logging.exception('Cannot convert %s', pathname)
    return collection


SECTION_TITLES = re.compile(r'('
                            r'ABDOMEN AND PELVIS|CLINICAL HISTORY|CLINICAL INDICATION|CLINICAL INFORMATION|COMPARISON|COMPARISON STUDY DATE|SUMMARY|REPORT'
                            r'|EXAM|EXAMINATION|FINDINGS|HISTORY|IMPRESSION|INDICATION|MRI WHOLE SPINE|CONCLUSION|CONCLUSIONS|MRI CERVICAL SPINE|MRI LUMBAR SPINE'
                            r'|MEDICAL CONDITION|PROCEDURE|REASON FOR EXAM|REASON FOR STUDY|REASON FOR THIS EXAMINATION|COMMENT|WHOLE SPINE|HISTORY'
                            r'|TECHNIQUE|MRI SPINE WHOLE|MRI SPINE CERVICAL|MRI SPINE|MRI SPINE THORACIC'
                            r'):|FINAL REPORT',
                            re.IGNORECASE | re.MULTILINE)

SECTION_LBREAK_TITLES = re.compile(r'('
                            r'ABDOMEN AND PELVIS|CLINICAL HISTORY|CLINICAL INDICATION|CLINICAL INFORMATION|COMPARISON|COMPARISON STUDY DATE|SUMMARY|REPORT'
                            r'|EXAM|EXAMINATION|FINDINGS|HISTORY|IMPRESSION|INDICATION|MRI WHOLE SPINE|CONCLUSION|CONCLUSIONS|MRI CERVICAL SPINE|MRI LUMBAR SPINE'
                            r'|MEDICAL CONDITION|PROCEDURE|REASON FOR EXAM|REASON FOR STUDY|REASON FOR THIS EXAMINATION|COMMENT|WHOLE SPINE'
                            r'|TECHNIQUE|MRI SPINE WHOLE|MRI SPINE CERVICAL|MRI SPINE'
                            r')\n|FINAL REPORT',
                            re.IGNORECASE | re.MULTILINE)

SECTION_LBREAK_TITLES2 = re.compile(r'('
                            r'ABDOMEN AND PELVIS|CLINICAL HISTORY|CLINICAL INDICATION|CLINICAL INFORMATION|COMPARISON|COMPARISON STUDY DATE|SUMMARY|REPORT'
                            r'|EXAM|EXAMINATION|FINDINGS|HISTORY|IMPRESSION|INDICATION|MRI WHOLE SPINE|CONCLUSION|CONCLUSIONS|MRI CERVICAL SPINE|MRI LUMBAR SPINE'
                            r'|MEDICAL CONDITION|PROCEDURE|REASON FOR EXAM|REASON FOR STUDY|REASON FOR THIS EXAMINATION|COMMENT|WHOLE SPINE'
                            r'|TECHNIQUE|MRI SPINE WHOLE|MRI SPINE CERVICAL|MRI SPINE'
                            r')\r\n|FINAL REPORT',
                            re.IGNORECASE | re.MULTILINE)

def is_empty(passage):
    return len(passage.text) == 0


def strip(passage):
    start = 0
    while start < len(passage.text) and passage.text[start].isspace():
        start += 1

    end = len(passage.text)
    while end > start and passage.text[end - 1].isspace():
        end -= 1

    passage.offset += start
    logging.debug('before: %r' % passage.text)
    passage.text = passage.text[start:end]
    logging.debug('after:  %r' % passage.text)
    return passage

def split_document(document, pattern=None):
    """
    Split one report into sections. Section splitting is a deterministic consequence of section titles.

    Args:
        document(BioCDocument): one document that contains one passage.
        pattern: the regular expression patterns for section titles.

    Returns:
        BioCDocument: a new BioCDocument instance
    """

    if pattern is None:
        pattern = SECTION_TITLES

    new_document = bioc.BioCDocument()
    new_document.id = document.id
    new_document.infons = document.infons

    text = document.passages[0].text
    offset = document.passages[0].offset

    def create_passage(start, end, title=None):
        passage = bioc.BioCPassage()
        passage.offset = start + offset
        passage.text = text[start:end]
        if title is not None:
            passage.infons['title'] = title[:-1].strip() if title[-1] == ':' else title.strip()
            passage.infons['type'] = 'title_1'
        strip(passage)
        return passage

    start = 0
    for matcher in pattern.finditer(text):
        logging.debug('Match: %s', matcher.group())
        # add last
        end = matcher.start()
        if end != start:
            passage = create_passage(start, end)
            if not is_empty(passage):
                new_document.add_passage(passage)

        start = end

        # add title
        end = matcher.end()
        passage = create_passage(start, end, text[start:end])
        if not is_empty(passage):
            new_document.add_passage(passage)

        start = end

    # add last piece
    end = len(text)
    if start < end:
        passage = create_passage(start, end)
        if not is_empty(passage):
            new_document.add_passage(passage)
    return new_document

def clean_doc(doc):
    # Patterns for confusing notes
    doc = doc.replace('AN ADDENDUM HAS BEEN ENTERED AT THE END OF THIS REPORT', '')
    # Patterns to fix for findings
    doc = doc.replace('\r\n\r\n\r\n', ' FINDINGS: ')
    doc = doc.replace('\r\n\r\n\r\n\r\n', ' FINDINGS: ')
    doc = doc.replace('\\r\\n\\r\\n', ' FINDINGS: ')
    doc = doc.replace('Findings;', ' FINDINGS: ')
    doc = doc.replace('Findings.', ' FINDINGS: ')
    doc = doc.replace('Report.', ' FINDINGS: ')
    doc = doc.replace('Metastatic spine protocol    ', 'FINDINGS: ')
    doc = doc.replace('Report Body MRI Spine cervical - ', 'FINDINGS: ')
    # Patterns to fix clinical history
    doc = doc.replace('Clinical history.', ' FINDINGS: ')
    doc = doc.replace('CLINICDET =', 'CLINICAL HISTORY: ')
    doc = doc.replace('MWSPN - MRI Whole Spine', 'CLINICAL HISTORY: ')
    doc = doc.replace('MWSPN -', 'CLINICAL HISTORY: ')
    if doc.find("b'") == 0:
        doc = doc[2:-1]
    doc = doc.replace(" :",":")
    for match in SECTION_LBREAK_TITLES.finditer(doc):
        doc = doc.replace(match[0], match[0].strip() + ': ')
    for match in SECTION_LBREAK_TITLES2.finditer(doc):
        doc = doc.replace(match[0], match[0].strip() + ': ')
    return doc

def merge_report(row):
    list_hist = ['CLINICAL INDICATION', 'Clinical History', 'CLINICAL HISTORY',
                 'Clinical history','Clinical indication', 'History','CLINICAL INFORMATION']
    li_new = []
    if len(row['seg_text'].keys()) == 1: 
        for key in row['seg_text']: 
            if key in list_hist:
                return row['report_text'].replace(key, '').replace(row['seg_text'][key], '')
    else: 
        for key in row['seg_text']:
            if (key not in list_hist):
                # If value (minus colon) not in headers, append
                if row['seg_text'][key][:-1] not in row['seg_text'].keys(): 
                    li_new.append(row['seg_text'][key])
        return '\n\n'.join(li_new).replace('[JOINED]','') # Remove joined tag

@hydra.main(version_base="1.2", config_path="../../conf", config_name="config")
def run_segmentation(cfg: DictConfig) -> None:

    df_test = pd.read_csv(cfg.full_ncimi_data, index_col=0)

    li_seg_reports = []
    for report in range(len(df_test)):
        bioc_doc = text2document(report, clean_doc(df_test['report_text'][report]))
        segment = split_document(bioc_doc)
        res = {}
        for i, passage in enumerate(segment.passages):
            if len(segment.passages) > 1: # test if there are identifiable sections
                if 'title' in passage.infons: # get section title
                    if passage.infons['title'] not in res: # if that section is new
                        try:
                            next_passage = segment.passages[i+1] # get content of section
                            res[passage.infons['title']] = next_passage.text # assign title to content
                        except:
                            pass
                    else: # if section already identified
                        try:
                            next_passage = segment.passages[i+1] 
                            res[passage.infons['title']] = res[passage.infons['title']] + " " + next_passage.text # add to the section content
                        except:
                            pass
                else: # move onto the next passage
                    pass
            else: # nothing identified – return full report
                res['Full Report'] = df_test['report_text'][report]
        li_seg_reports.append(res)

    seg_reports = pd.DataFrame(
        {'study_id_coded': df_test['study_id_coded'],
        'report_text': df_test['report_text'],
        'seg_text': li_seg_reports,})

    # Append all sections except clinical history
    seg_reports['report_no_hist'] = seg_reports.apply(lambda x: merge_report(x), axis=1)

    # Get only proper segmentations (length > 100)
    seg_reports = seg_reports.loc[seg_reports.report_no_hist.notnull()]
    seg_reports['seg_length'] = seg_reports['report_no_hist'].apply(lambda x: len(x))
    seg_reports = seg_reports.loc[seg_reports['seg_length']>100].reset_index(drop=True)
    
    seg_reports.to_csv(f'{cfg.updated_clean_path}/segmented_unique_reports.csv')

if __name__ == "__main__":
    run_segmentation()