import spacy
import tokenizations
import transformers

sent = "John didn't commit a crime."
transforms = [0, 0, 1, 0, 0, 0, 0]

nlp = spacy.load("en_core_web_sm")
tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-cased')

sent_nlp = nlp(sent)
spacy_tokens = [token.text for token in sent_nlp]
print(spacy_tokens)

bert_encoding = tokenizer.encode(sent)
bert_tokens = tokenizer.convert_ids_to_tokens(bert_encoding)
print(bert_tokens)

spacy_to_bert, _ = tokenizations.get_alignments(spacy_tokens, bert_tokens)
print(spacy_to_bert)
new_transforms = [0] * len(bert_tokens)

for spacy_ind, alignment in enumerate(spacy_to_bert):
    for new_ind in alignment:
        new_transforms[new_ind] = transforms[spacy_ind]

print(new_transforms)
