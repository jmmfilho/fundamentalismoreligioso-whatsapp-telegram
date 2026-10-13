PY ?= python

.PHONY: setup data datasets ner retrieval polarity topics liwc verify all test

setup:            ## install dependencies and language resources
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .
	$(PY) -m spacy download pt_core_news_sm
	$(PY) -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

data:             ## download the Git LFS datasets
	git lfs install && git lfs pull

datasets:  ; $(PY) -m fundrel datasets
ner:       ; $(PY) -m fundrel ner
retrieval: ; $(PY) -m fundrel retrieval
polarity:  ; $(PY) -m fundrel polarity
topics:    ; $(PY) -m fundrel topics
liwc:      ; $(PY) -m fundrel liwc
verify:    ; $(PY) -m fundrel verify

all:       ; $(PY) -m fundrel all

test:      ; $(PY) -m pytest -q
