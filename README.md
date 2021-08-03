# Supplementary Material for "Property-based Test for Part-of-Speech Tagging Tool"

## Implementation and Instruction

We implement our property-based test on three common POS tagging tools (e.g., spaCy, NLTK, and Flair) with a series of python scripts.

*All the data are stored in `MRs.zip`.*
#### (the OntoNotes5.0 dataset can be downloaded from https://catalog.ldc.upenn.edu/LDC2013T19)

---
### Usage Instruction to realize *four novel MRs*

1. run `MR1-1.py` to execute 100 tests with MR1-1 on three POS tagging tools
```bash
python MR1-1.py \
--tool_name name/of/tool/under/test \
--ontonotes5_test_file_path path/to/ontonotes5.0/test/file
--follow_up_inputs_csv_file_path path/to/follow/up/inputs/file
--violations_csv_file_path path/to/output/violations/file
```
2. run `MR1-2.py` to execute 100 tests with MR1-2 on three POS tagging tools
```bash
python MR1-2.py \
--tool_name name/of/tool/under/test \
--ontonotes5_test_file_path path/to/ontonotes5.0/test/file
--follow_up_inputs_csv_file_path path/to/follow/up/inputs/file
--violations_csv_file_path path/to/output/violations/file
```
3. run `MR2-1.py` to execute 100 tests with MR2-1 on three POS tagging tools
```bash
python MR2-1.py \
--tool_name name/of/tool/under/test \
--ontonotes5_test_file_path path/to/ontonotes5.0/test/file
--follow_up_inputs_csv_file_path path/to/follow/up/inputs/file
--violations_csv_file_path path/to/output/violations/file
```
4. run `MR2-2.py` to execute 100 tests with MR2-2 on three POS tagging tools
```bash
python MR2-2.py \
--tool_name name/of/tool/under/test \
--ontonotes5_test_file_path path/to/ontonotes5.0/test/file
--follow_up_inputs_csv_file_path path/to/follow/up/inputs/file
--violations_csv_file_path path/to/output/violations/file
```
---
## Detailed Test Results

We provide the raw test result files.

*All the data are stored in `results.zip`.*

---

### Content Structure
```
results
┝━━ flair
│   ┝━━ MR1-1
│   │      follow_input.csv: follow-up inputs of 100 eligible test case.
│   │      violation.csv: detail information of the violated cases.
│   ┝━━ MR1-2: (content structure is similar to above.)
│   ┝━━ MR2-1: (content structure is similar to above.)
│   ┕━━ MR2-2: (content structure is similar to above.)
┝━━ spacy: (content structure is similar to above.)
┕━━ nltk: (content structure is similar to above.)
```
---