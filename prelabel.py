import spacy
import tokenizations
import pandas as pd
from transformers import BertTokenizerFast
from pandarallel import pandarallel

# https://spacy.io/api/token

# Example dependency parser: https://corenlp.run

# https://github.com/explosion/spaCy/blob/master/spacy/glossary.py List of spacy tags

# https://towardsdatascience.com/natural-language-processing-dependency-parsing-cf094bbbe3f7
# Switched to spacy's "en_core_web_sm" parser instead of spacy–udpipe's "en" parser;
# latter was causing issues such as parsing "plans" in "John plans to commit a crime" as a noun.

# How to align spaCy and BERT tokens: https://gist.github.com/tamuhey/af6cbb44a703423556c32798e1e1b704
# Now hosted here: https://github.com/explosion/tokenizations

# --- Transform markers ---
NO_TRANSFORM =        0 # John commits a crime.
MODAL_TRANSFORM =     1 # John must commit a crime.
INTENTION_TRANSFORM = 2 # John plans to commit a crime.
RESULT_TRANSFORM = 	  3 # John failed to commit a crime.
ASPECT_TRANSFORM =    4 # John starts to commit a crime.
STATUS_TRANSFORM =    5 # John did not commit a crime.
MANNER_TRANSFORM =    6 # John is eager to commit a crime.

INTENTION_LEMMAS = ["plan", "hope", "intend", "try", "premeditate", "aim", "propose", "aspire", "hope", "expect", "want", "contemplate", "envision", "envisage"]
RESULT_LEMMAS = ["succeed", "manage", "obtain", "fail", "accomplish", "achieve"]
ASPECT_LEMMAS = ["begin", "start", "finish", "end", "initiate", "commence", "continue", "persist"]
STATUS_LEMMAS = ["not"]

REVIEW_CSV_PATH = 'data/unlabeled-training-data.csv'
BIN_OUTPUT_CSV_PATH = 'data/bin-prelabeled.csv'
MM_OUTPUT_CSV_PATH = 'data/mm-prelabeled.csv'

TOKENIZER_MODEL = 'bert-base-cased'

INSTANCE_COUNT = 100_000

def isModal(token) -> bool:
	return token.tag_ == "MD" and token.head.pos_ == "VERB"

def isIntention(token) -> bool:
	if token.pos_ == "VERB" and token.lemma_ in INTENTION_LEMMAS:
		for child in token.children:
			if child.dep_ == "xcomp":
				return True
	return False

def isResult(token) -> bool:
	return token.head.pos_ == "VERB" and token.lemma_ in RESULT_LEMMAS

def isAspect(token) -> bool:
	return token.pos_ == "VERB" and token.lemma_ in ASPECT_LEMMAS

def isStatus(token) -> bool:
	return token.head.pos_ in ["VERB", "AUX"] and token.lemma_ in STATUS_LEMMAS

def markMMTransforms(sentence) -> str:
	mm_transforms = [NO_TRANSFORM] * len(sentence)
	for token_ind, token in enumerate(sentence):
		if isModal(token):
			mm_transforms[token_ind] = MODAL_TRANSFORM
		elif isIntention(token):
			mm_transforms[token_ind] = INTENTION_TRANSFORM
		elif isResult(token):
			mm_transforms[token_ind] = RESULT_TRANSFORM
		elif isAspect(token):
			mm_transforms[token_ind] = ASPECT_TRANSFORM
		elif isStatus(token):
			mm_transforms[token_ind] = STATUS_TRANSFORM
		elif token.pos_ in ["VERB", "AUX"]:
			# process manners in one pass
			for child in token.children:
				if child.pos_ in ["ADJ", "ADV"]:
					for subchild in child.children:
						if subchild.pos_ == "VERB":
							mm_transforms[child.i] = MANNER_TRANSFORM
							break
	return mm_transforms

def splitTokens(text):
	doc = nlp(text)
	if len(list(doc.sents)) > 1:
		return ''
	return doc

def mapTransformTokens(doc, bert_tokens, mm_transforms):
	spacy_tokens = [t.text for t in doc]
	mapped_transforms = [0] * len(bert_tokens)
	spacy_to_bert, _ = tokenizations.get_alignments(spacy_tokens, bert_tokens)
	for spacy_ind, al in enumerate(spacy_to_bert):
		for bert_ind in al:
			mapped_transforms[bert_ind] = mm_transforms[spacy_ind]
	return mapped_transforms

def processTransforms(df):
	print("Performing spaCy tokenization...")
	df['doc'] = df['instance'].parallel_apply(splitTokens) # nlp object / doc
	print("Performing BERT tokenization...")
	df['bert_tokens'] = df['instance'].parallel_apply(lambda x: tokenizer.encode(x)) # list of integers
	print("Converting BERT tokens to strings...")
	df['bert_tokens'] = df['bert_tokens'].parallel_apply(lambda x: tokenizer.convert_ids_to_tokens(x)) # list of strings
	print("Performing rule-based transform marking...")
	df['mm_transforms'] = df['doc'].parallel_apply(markMMTransforms) # list of integers
	print("Pruning empty rows...")
	df = df[df['mm_transforms'].map(bool)] # remove rows where the mm_transforms list is empty: https://stackoverflow.com/questions/34162625/remove-rows-with-empty-lists-from-pandas-data-frame
	print("Mapping spaCy tokens to BERT tokens...")
	df['mm_transforms'] = df.parallel_apply(lambda x: mapTransformTokens(x.doc, x.bert_tokens, x.mm_transforms), axis=1) # list of integers
	print("Extracting MM transforms...")
	df['mm_transforms'] = df['mm_transforms'].parallel_apply(lambda x: ' '.join(str(y) for y in x)) # convert list to string
	print("Removing ")
	print("Extracting binary transforms...")
	df['bin_transforms'] = df['mm_transforms'].parallel_apply(lambda x: ' '.join('0' if y == '0' else '1' for y in x.split(' '))) # convert mm string to bin string
	return df

def test(sentence: str):
	doc = nlp(sentence)
	dep_sents = [str(x) for x in list(doc.sents)]
	df = pd.DataFrame(dep_sents, columns=['instance'])
	df = processTransforms(df)
	for i in df.index:
		print()
		print(df['instance'][i])
		print(df['mm_transforms'][i])
	exit()

if __name__ == '__main__':	
	pandarallel.initialize(progress_bar=True, verbose=False)
	tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_MODEL)
	nlp = spacy.load("en_core_web_sm")

	# test("John did not manage to commit a crime. John does not plan to commit a crime. John did not continue eating. John is not ready to commit a crime. John, despite his prior enthusiasm, is not eager to commit a crime.")

	inputDF = pd.read_csv(REVIEW_CSV_PATH)
	inputDF = inputDF.head(INSTANCE_COUNT)
	
	print(f"Processing {len(inputDF)} lines.")
	reviewDF = processTransforms(inputDF)
	
	binDF = reviewDF[['review-id', 'instance', 'bin_transforms', 'file-name']]
	mmDF = reviewDF[['review-id', 'instance', 'mm_transforms', 'file-name']]
	
	print(f"Writing to {BIN_OUTPUT_CSV_PATH}...")
	binDF.to_csv(BIN_OUTPUT_CSV_PATH)
	
	print(f"Writing to {MM_OUTPUT_CSV_PATH}...")
	mmDF.to_csv(MM_OUTPUT_CSV_PATH)
	
	print("Finished!")