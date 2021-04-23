# coding: utf-8
# Author: C.J. Hutto
# Thanks to George Berry for reducing the time complexity from something like O(N^4) to O(N).
# Thanks to Ewan Klein and Pierpaolo Pantone for bringing VADER into NLTK. Those modifications were awesome.
# For license information, see LICENSE.TXT

# Adapted for Portuguese by Fernando Batista (fernando.batista.pt @ gmail.com)

"""
If you use the VADER sentiment analysis tools, please cite:
Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for
Sentiment Analysis of Social Media Text. Eighth International Conference on
Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
"""
import os
import re
import math
import string
import codecs
import json
from itertools import product
from inspect import getsourcefile
from io import open

def normalize(score, alpha=15):
    """
    Normalize the score to be between -1 and 1 using an alpha that
    approximates the max expected value
    """
    norm_score = score / math.sqrt((score * score) + alpha)
    if norm_score < -1.0:
        return -1.0
    elif norm_score > 1.0:
        return 1.0
    else:
        return norm_score

def allcap_differential(words):
    """
    Check whether just some words in the input are ALL CAPS
    :param list words: The words to inspect
    :returns: `True` if some but not all items in `words` are ALL CAPS
    """
    is_different = False
    allcap_words = 0
    for word in words:
        if word.isupper():
            allcap_words += 1
    cap_differential = len(words) - allcap_words
    if 0 < cap_differential < len(words):
        is_different = True
    return is_different


class LanguageKB():
    
    def __init__(self, language, B_INCR, B_DECR, C_INCR, N_SCALAR):
        
        assert(language in ["en", "pt"])
        self.B_INCR = B_INCR
        self.C_INCR = C_INCR
        self.N_SCALAR = N_SCALAR
        self.B_DECR = B_DECR
        self.language = language
        
        self.NEGATE = self.read_negate(f"./negate.{language}.txt")
        # booster/dampener 'intensifiers' or 'degree adverbs'
        # http://en.wiktionary.org/wiki/Category:English_degree_adverbs
        self.BOOSTER_DICT = self.read_booster_dict(f"./booster-dict.{language}.txt", B_INCR, B_DECR)
        # check for sentiment laden idioms that do not contain lexicon words (future work, not yet implemented)
        self.SENTIMENT_LADEN_IDIOMS = self.read_idioms(f"./idioms.{language}.txt")
        # check for special case idioms and phrases containing lexicon words
        self.SPECIAL_CASES = self.read_special_cases(f"./special-cases.{language}.txt")
        self.LEXICON = self.read_lexicon(f"./vader_lexicon.{language}.txt")
        self.EMOJIS = self.read_emoji("./emoji_utf8_lexicon.txt")
            
        if language == "en":
            self.WORD = {'no':'no', 'kind':'kind', 'of':'of', 'at':'at', 'least':'least', 'very':'very', 'but':'but', 
                 'never':'never', 'so': 'so', 'this': 'this', 'without':'without', 'doubt':'doubt', 'or':'or', 'nor':'nor'}
        elif language == "pt":
            self.WORD = {'no':'não', 'kind':'tipo', 'of':'de', 'at':'pelo', 'least':'menos', 'very':'muito', 'but':'mas', 
                 'never':'nunca', 'so': 'assim', 'this': 'isto', 'without':'sem', 'doubt':'dúvida', 'or':'ou', 'nor': 'nem'}
            
    @staticmethod
    def read_lexicon(filename):
        """
        Convert lexicon file to a dictionary
        """
        lex_dict = {}
        with open(filename, encoding="utf-8") as myfile:
            for line in myfile:
                if line.startswith("#") or len(line.strip()) == 0:
                    continue            
                (word, measure) = line.strip().split("\t")[0:2]
                lex_dict[word] = float(measure)
        return lex_dict
    
    @staticmethod
    def read_emoji(filename):
        """
        Convert emoji lexicon file to a dictionary
        """
        emoji_dict = {}
        with open(filename, encoding="utf-8") as myfile:
            for line in myfile:
                if line.startswith("#") or len(line.strip()) == 0:
                    continue            
                (emoji, description) = line.strip().split("\t")[0:2]
                emoji_dict[emoji] = description
        return emoji_dict

    @staticmethod
    def read_negate(filename):
        with open(filename, encoding="utf-8") as myfile:
            negate = [line.strip() for line in myfile]
            return negate

    @staticmethod
    def read_booster_dict(filename, B_INCR, B_DECR):
        booster = {}
        with open(filename, encoding="utf-8") as myfile:
            for line in myfile:
                if line.startswith("#") or len(line.strip()) == 0:
                    continue            
                info = line.strip().split("\t")
                if info[1] == "1":
                    booster[info[0]] = B_INCR
                elif info[1] == "-1":
                    booster[info[0]] = B_DECR
        return booster

    @staticmethod
    def read_idioms(filename):
        idioms = {}
        with open(filename, encoding="utf-8") as myfile:
            for line in myfile:
                if line.startswith("#") or len(line.strip()) == 0:
                    continue            
                info = line.strip().split("\t")
                idioms[info[0]] = int(info[1])
        return idioms

    @staticmethod
    def read_special_cases(filename):
        scases = {}
        with open(filename, encoding="utf-8") as myfile:
            for line in myfile:
                if line.startswith("#") or len(line.strip()) == 0:
                    continue
                info = line.strip().split("\t")
                scases[info[0]] = float(info[1])
        return scases

    def negated(self, input_words, include_nt=True):
        """
        Determine if input contains negation words
        """
        input_words = [str(w).lower() for w in input_words]
        neg_words = []
        neg_words.extend(self.NEGATE)
        for word in neg_words:
            if word in input_words:
                return True
        if include_nt:
            for word in input_words:
                if "n't" in word:
                    return True
        '''if "least" in input_words:
            i = input_words.index("least")
            if i > 0 and input_words[i - 1] != "at":
                return True'''
        return False

    def scalar_inc_dec(self, word, valence, is_cap_diff):
        """
        Check if the preceding words increase, decrease, or negate/nullify the
        valence
        """
        scalar = 0.0
        word_lower = word.lower()
        if word_lower in self.BOOSTER_DICT:
            scalar = self.BOOSTER_DICT[word_lower]
            if valence < 0:
                scalar *= -1
            # check if booster/dampener word is in ALLCAPS (while others aren't)
            if word.isupper() and is_cap_diff:
                if valence > 0:
                    scalar += self.C_INCR
                else:
                    scalar -= self.C_INCR
        return scalar

class SentiText(object):
    """
    Identify sentiment-relevant string-level properties of input text.
    """

    def __init__(self, text):
        if not isinstance(text, str):
            text = str(text).encode('utf-8')
        self.text = text
        self.words_and_emoticons = self._words_and_emoticons()
        # doesn't separate words from\
        # adjacent punctuation (keeps emoticons & contractions)
        self.is_cap_diff = allcap_differential(self.words_and_emoticons)

    @staticmethod
    def _strip_punc_if_word(token):
        """
        Removes all trailing and leading punctuation
        If the resulting string has two or fewer characters,
        then it was likely an emoticon, so return original string
        (ie ":)" stripped would be "", so just return ":)"
        """
        stripped = token.strip(string.punctuation)
        if len(stripped) <= 2:
            return token
        return stripped

    def _words_and_emoticons(self):
        """
        Removes leading and trailing puncutation
        Leaves contractions and most emoticons
            Does not preserve punc-plus-letter emoticons (e.g. :D)
        """
        wes = self.text.split()
        stripped = list(map(self._strip_punc_if_word, wes))
        return stripped

    
class SentimentIntensityAnalyzer(object):
    """
    Give a sentiment intensity score to sentences.
    """

    def __init__(self, language = "en"):

        # Constants
        # B_INCR, B_DECR: empirically derived mean sentiment intensity rating increase for booster words
        # C_INCR, N_SCALAR: empirically derived mean sentiment intensity rating increase for using ALLCAPs to emphasize a word
        self.KB = LanguageKB(language, B_INCR=0.293, B_DECR=-0.293, C_INCR=0.733, N_SCALAR=-0.74)
        
        
    def polarity_scores(self, text):
        """
        Return a float for sentiment strength based on the input text.
        Positive values are positive valence, negative value are negative
        valence.
        """
        # convert emojis to their textual descriptions
        text_no_emoji = ""
        prev_space = True
        for chr in text:
            if chr in self.KB.EMOJIS:
                # get the textual description
                description = self.KB.EMOJIS[chr]
                if not prev_space:
                    text_no_emoji += ' '
                text_no_emoji += description
                prev_space = False
            else:
                text_no_emoji += chr
                prev_space = chr == ' '
        text = text_no_emoji.strip()

        sentitext = SentiText(text)

        sentiments = []
        words_and_emoticons = sentitext.words_and_emoticons
        for i, item in enumerate(words_and_emoticons):
            valence = 0
            # check for vader_lexicon words that may be used as modifiers or negations
            if item.lower() in self.KB.BOOSTER_DICT:
                sentiments.append(valence)
                continue
            if (i < len(words_and_emoticons) - 1 and item.lower() == self.KB.WORD["kind"] and
                    words_and_emoticons[i + 1].lower() == self.KB.WORD["of"]):
                sentiments.append(valence)
                continue

            sentiments = self.sentiment_valence(valence, sentitext, item, i, sentiments)

        sentiments = self._but_check(words_and_emoticons, sentiments)

        valence_dict = self.score_valence(sentiments, text)

        return valence_dict
    
    def sentiment_valence(self, valence, sentitext, item, i, sentiments):
        is_cap_diff = sentitext.is_cap_diff
        words_and_emoticons = sentitext.words_and_emoticons
        item_lowercase = item.lower()
        if item_lowercase in self.KB.LEXICON:
            # get the sentiment valence 
            valence = self.KB.LEXICON[item_lowercase]
                
            # check for "no" as negation for an adjacent lexicon item vs "no" as its own stand-alone lexicon item
            if item_lowercase == self.KB.WORD["no"] and i != len(words_and_emoticons)-1 and words_and_emoticons[i + 1].lower() in self.KB.LEXICON:
                # don't use valence of "no" as a lexicon item. Instead set it's valence to 0.0 and negate the next item
                valence = 0.0
            if (i > 0 and words_and_emoticons[i - 1].lower() == self.KB.WORD["no"]) \
               or (i > 1 and words_and_emoticons[i - 2].lower() == self.KB.WORD["no"]) \
               or (i > 2 and words_and_emoticons[i - 3].lower() == self.KB.WORD["no"] \
                   and words_and_emoticons[i - 1].lower() in [self.KB.WORD["or"], self.KB.WORD["nor"]] ):
                valence = self.KB.LEXICON[item_lowercase] * self.KB.N_SCALAR
            
            # check if sentiment laden word is in ALL CAPS (while others aren't)
            if item.isupper() and is_cap_diff:
                if valence > 0:
                    valence += self.KB.C_INCR
                else:
                    valence -= self.KB.C_INCR

            for start_i in range(0, 3):
                # dampen the scalar modifier of preceding words and emoticons
                # (excluding the ones that immediately preceed the item) based
                # on their distance from the current item.
                if i > start_i and words_and_emoticons[i - (start_i + 1)].lower() not in self.KB.LEXICON:
                    s = self.KB.scalar_inc_dec(words_and_emoticons[i - (start_i + 1)], valence, is_cap_diff)
                    if start_i == 1 and s != 0:
                        s = s * 0.95
                    if start_i == 2 and s != 0:
                        s = s * 0.9
                    valence = valence + s
                    valence = self._negation_check(valence, words_and_emoticons, start_i, i)
                    if start_i == 2:
                        valence = self._special_idioms_check(valence, words_and_emoticons, i)

            valence = self._least_check(valence, words_and_emoticons, i)
        sentiments.append(valence)
        return sentiments

    def _least_check(self, valence, words_and_emoticons, i):
        # check for negation case using "least"
        if i > 1 and words_and_emoticons[i - 1].lower() not in self.KB.LEXICON \
                and words_and_emoticons[i - 1].lower() == self.KB.WORD["least"]:
            if words_and_emoticons[i - 2].lower() != self.KB.WORD["at"] and words_and_emoticons[i - 2].lower() != self.KB.WORD["very"]:
                valence = valence * self.KB.N_SCALAR
        elif i > 0 and words_and_emoticons[i - 1].lower() not in self.KB.LEXICON \
                and words_and_emoticons[i - 1].lower() == self.KB.WORD["least"]:
            valence = valence * self.KB.N_SCALAR
        return valence

    def _but_check(self, words_and_emoticons, sentiments):
        # check for modification in sentiment due to contrastive conjunction 'but'
        words_and_emoticons_lower = [str(w).lower() for w in words_and_emoticons]
        if self.KB.WORD['but'] in words_and_emoticons_lower:
            bi = words_and_emoticons_lower.index(self.KB.WORD['but'])
            for sentiment in sentiments:
                si = sentiments.index(sentiment)
                if si < bi:
                    sentiments.pop(si)
                    sentiments.insert(si, sentiment * 0.5)
                elif si > bi:
                    sentiments.pop(si)
                    sentiments.insert(si, sentiment * 1.5)
        return sentiments

    def _special_idioms_check(self, valence, words_and_emoticons, i):
        words_and_emoticons_lower = [str(w).lower() for w in words_and_emoticons]
        onezero = "{0} {1}".format(words_and_emoticons_lower[i - 1], words_and_emoticons_lower[i])

        twoonezero = "{0} {1} {2}".format(words_and_emoticons_lower[i - 2],
                                          words_and_emoticons_lower[i - 1], words_and_emoticons_lower[i])

        twoone = "{0} {1}".format(words_and_emoticons_lower[i - 2], words_and_emoticons_lower[i - 1])

        threetwoone = "{0} {1} {2}".format(words_and_emoticons_lower[i - 3],
                                           words_and_emoticons_lower[i - 2], words_and_emoticons_lower[i - 1])

        threetwo = "{0} {1}".format(words_and_emoticons_lower[i - 3], words_and_emoticons_lower[i - 2])

        sequences = [onezero, twoonezero, twoone, threetwoone, threetwo]

        for seq in sequences:
            if seq in self.KB.SPECIAL_CASES:
                valence = self.KB.SPECIAL_CASES[seq]
                break

        if len(words_and_emoticons_lower) - 1 > i:
            zeroone = "{0} {1}".format(words_and_emoticons_lower[i], words_and_emoticons_lower[i + 1])
            if zeroone in self.KB.SPECIAL_CASES:
                valence = self.KB.SPECIAL_CASES[zeroone]
        if len(words_and_emoticons_lower) - 1 > i + 1:
            zeroonetwo = "{0} {1} {2}".format(words_and_emoticons_lower[i], words_and_emoticons_lower[i + 1],
                                              words_and_emoticons_lower[i + 2])
            if zeroonetwo in self.KB.SPECIAL_CASES:
                valence = self.KB.SPECIAL_CASES[zeroonetwo]

        # check for booster/dampener bi-grams such as 'sort of' or 'kind of'
        n_grams = [threetwoone, threetwo, twoone]
        for n_gram in n_grams:
            if n_gram in self.KB.BOOSTER_DICT:
                valence = valence + self.KB.BOOSTER_DICT[n_gram]
        return valence

    @staticmethod
    def _sentiment_laden_idioms_check(valence, senti_text_lower):
        # Future Work
        # check for sentiment laden idioms that don't contain a lexicon word
        idioms_valences = []
        for idiom in self.KB.SENTIMENT_LADEN_IDIOMS:
            if idiom in senti_text_lower:
                print(idiom, senti_text_lower)
                valence = self.KB.SENTIMENT_LADEN_IDIOMS[idiom]
                idioms_valences.append(valence)
        if len(idioms_valences) > 0:
            valence = sum(idioms_valences) / float(len(idioms_valences))
        return valence

    def _negation_check(self, valence, words_and_emoticons, start_i, i):
        words_and_emoticons_lower = [str(w).lower() for w in words_and_emoticons]
        if start_i == 0:
            if self.KB.negated([words_and_emoticons_lower[i - (start_i + 1)]]):  # 1 word preceding lexicon word (w/o stopwords)
                valence = valence * self.KB.N_SCALAR
        if start_i == 1:
            if words_and_emoticons_lower[i - 2] == self.KB.WORD["never"] and \
                    (words_and_emoticons_lower[i - 1] in [self.KB.WORD["so"], self.KB.WORD["this"]]):
                valence = valence * 1.25
            elif words_and_emoticons_lower[i - 2] == self.KB.WORD["without"] and \
                    words_and_emoticons_lower[i - 1] == self.KB.WORD["doubt"]:
                valence = valence
            elif self.KB.negated([words_and_emoticons_lower[i - (start_i + 1)]]):  # 2 words preceding the lexicon word position
                valence = valence * self.KB.N_SCALAR
        if start_i == 2:
            if words_and_emoticons_lower[i - 3] == self.KB.WORD["never"] and \
                    (words_and_emoticons_lower[i - 2] in [self.KB.WORD["so"], self.KB.WORD["this"]]) or \
                    (words_and_emoticons_lower[i - 1] in [self.KB.WORD["so"], self.KB.WORD["this"]]):
                valence = valence * 1.25
            elif words_and_emoticons_lower[i - 3] == self.KB.WORD["without"] and \
                    (words_and_emoticons_lower[i - 2] == self.KB.WORD["doubt"] or words_and_emoticons_lower[i - 1] == self.KB.WORD["doubt"]):
                valence = valence
            elif self.KB.negated([words_and_emoticons_lower[i - (start_i + 1)]]):  # 3 words preceding the lexicon word position
                valence = valence * self.KB.N_SCALAR
        return valence

    def _punctuation_emphasis(self, text):
        # add emphasis from exclamation points and question marks
        ep_amplifier = self._amplify_ep(text)
        qm_amplifier = self._amplify_qm(text)
        punct_emph_amplifier = ep_amplifier + qm_amplifier
        return punct_emph_amplifier

    @staticmethod
    def _amplify_ep(text):
        # check for added emphasis resulting from exclamation points (up to 4 of them)
        ep_count = text.count("!")
        if ep_count > 4:
            ep_count = 4
        # (empirically derived mean sentiment intensity rating increase for
        # exclamation points)
        ep_amplifier = ep_count * 0.292
        return ep_amplifier

    @staticmethod
    def _amplify_qm(text):
        # check for added emphasis resulting from question marks (2 or 3+)
        qm_count = text.count("?")
        qm_amplifier = 0
        if qm_count > 1:
            if qm_count <= 3:
                # (empirically derived mean sentiment intensity rating increase for
                # question marks)
                qm_amplifier = qm_count * 0.18
            else:
                qm_amplifier = 0.96
        return qm_amplifier

    @staticmethod
    def _sift_sentiment_scores(sentiments):
        # want separate positive versus negative sentiment scores
        pos_sum = 0.0
        neg_sum = 0.0
        neu_count = 0
        for sentiment_score in sentiments:
            if sentiment_score > 0:
                pos_sum += (float(sentiment_score) + 1)  # compensates for neutral words that are counted as 1
            if sentiment_score < 0:
                neg_sum += (float(sentiment_score) - 1)  # when used with math.fabs(), compensates for neutrals
            if sentiment_score == 0:
                neu_count += 1
        return pos_sum, neg_sum, neu_count

    def score_valence(self, sentiments, text):
        if sentiments:
            sum_s = float(sum(sentiments))
            # compute and add emphasis from punctuation in text
            punct_emph_amplifier = self._punctuation_emphasis(text)
            if sum_s > 0:
                sum_s += punct_emph_amplifier
            elif sum_s < 0:
                sum_s -= punct_emph_amplifier

            compound = normalize(sum_s)
            # discriminate between positive, negative and neutral sentiment scores
            pos_sum, neg_sum, neu_count = self._sift_sentiment_scores(sentiments)

            if pos_sum > math.fabs(neg_sum):
                pos_sum += punct_emph_amplifier
            elif pos_sum < math.fabs(neg_sum):
                neg_sum -= punct_emph_amplifier

            total = pos_sum + math.fabs(neg_sum) + neu_count
            pos = math.fabs(pos_sum / total)
            neg = math.fabs(neg_sum / total)
            neu = math.fabs(neu_count / total)

        else:
            compound = 0.0
            pos = 0.0
            neg = 0.0
            neu = 0.0

        sentiment_dict = \
            {"neg": round(neg, 3),
             "neu": round(neu, 3),
             "pos": round(pos, 3),
             "compound": round(compound, 4)}

        return sentiment_dict


if __name__ == '__main__':
    lang = 'pt'
    sentences={}
    sentences['en'] = ["VADER is smart, handsome, and funny.",  # positive sentence example
                 "VADER is smart, handsome, and funny!",
                 # punctuation emphasis handled correctly (sentiment intensity adjusted)
                 "VADER is very smart, handsome, and funny.",
                 # booster words handled correctly (sentiment intensity adjusted)
                 "VADER is VERY SMART, handsome, and FUNNY.",  # emphasis for ALLCAPS handled
                 "VADER is VERY SMART, handsome, and FUNNY!!!",
                 # combination of signals - VADER appropriately adjusts intensity
                 "VADER is VERY SMART, uber handsome, and FRIGGIN FUNNY!!!",
                 # booster words & punctuation make this close to ceiling for score
                 "VADER is not smart, handsome, nor funny.",  # negation sentence example
                 "The book was good.",  # positive sentence
                 "At least it isn't a horrible book.",  # negated negative sentence with contraction
                 "The book was only kind of good.",
                 # qualified positive sentence is handled correctly (intensity adjusted)
                 "The plot was good, but the characters are uncompelling and the dialog is not great.",
                 # mixed negation sentence
                 "Today SUX!",  # negative slang with capitalization emphasis
                 "Today only kinda sux! But I'll get by, lol",
                 # mixed sentiment example with slang and constrastive conjunction "but"
                 "Make sure you :) or :D today!",  # emoticons handled
                 "Catch utf-8 emoji such as 💘 and 💋 and 😁",  # emojis handled
                 "Not bad at all"  # Capitalized negation
                 ]
    sentences['pt'] = ["VADER é inteligente, bonito e engraçado", # exemplo de uma frase positiva
                "O VADER é esperto, bonito e engraçado",
                "O VADER é muito esperto, bonito e engraçado",
                "VADER é MUITO ESPERTO, bonito, e divertido", # ênfase para ALLCAPS tratado
                "VADER é MUITO INTELIGENTE, bonito, e DIVERTIDO!!!",
                "VADER é MUITO INTELIGENTE, bonito, e SUPER ENGRAÇADO!!!",
                "VADER não é inteligente, bonito, nem engraçado", # exemplo de frase de negação
                "O livro era bom", # frase positiva
                "Pelo menos não é um livro horrível", # negou a frase negativa com contracção
                "O livro era apenas do tipo bom.",
                "O enredo foi bom, mas as personagens eram péssimas e o diálogo não era excelente.",
                "Hoje está de chuva!", # calão negativo com ênfase na capitalização
                "Hoje está mauzito! Mas eu vou sobreviver, lol",
                "Assegura-te de que :) ou :D hoje!", # emoticons handled
                "Apanhar emoji utf-8 tais como 💘 e 💋 e 😁", # emojis tratados
                "Nada mau, não é ?" # Negação capitalizada
                ]
                                                            

    analyzer = SentimentIntensityAnalyzer(lang)

    print("----------------------------------------------------")
    print(" - Analyze typical example cases, including handling of:")
    print("  -- negations")
    print("  -- punctuation emphasis & punctuation flooding")
    print("  -- word-shape as emphasis (capitalization difference)")
    print("  -- degree modifiers (intensifiers such as 'very' and dampeners such as 'kind of')")
    print("  -- slang words as modifiers such as 'uber' or 'friggin' or 'kinda'")
    print("  -- contrastive conjunction 'but' indicating a shift in sentiment; sentiment of later text is dominant")
    print("  -- use of contractions as negations")
    print("  -- sentiment laden emoticons such as :) and :D")
    print("  -- utf-8 encoded emojis such as 💘 and 💋 and 😁")
    print("  -- sentiment laden slang words (e.g., 'sux')")
    print("  -- sentiment laden initialisms and acronyms (for example: 'lol') \n")
    for sentence in sentences[lang]:
        vs = analyzer.polarity_scores(sentence)
        print("{:-<65} {}".format(sentence, str(vs)))
    print("----------------------------------------------------")
    print(" - About the scoring: ")
    print("""  -- The 'compound' score is computed by summing the valence scores of each word in the lexicon, adjusted
     according to the rules, and then normalized to be between -1 (most extreme negative) and +1 (most extreme positive).
     This is the most useful metric if you want a single unidimensional measure of sentiment for a given sentence.
     Calling it a 'normalized, weighted composite score' is accurate.""")
    print("""  -- The 'pos', 'neu', and 'neg' scores are ratios for proportions of text that fall in each category (so these
     should all add up to be 1... or close to it with float operation).  These are the most useful metrics if
     you want multidimensional measures of sentiment for a given sentence.""")
    print("----------------------------------------------------")

    input("\nPress Enter to continue the demo...\n")  # for DEMO purposes...

    tricky_sentences = ["Sentiment analysis has never been good.",
                        "Sentiment analysis has never been this good!",
                        "Most automated sentiment analysis tools are shit.",
                        "With VADER, sentiment analysis is the shit!",
                        "Other sentiment analysis tools can be quite bad.",
                        "On the other hand, VADER is quite bad ass",
                        "VADER is such a badass!",  # slang with punctuation emphasis
                        "Without a doubt, excellent idea.",
                        "Roger Dodger is one of the most compelling variations on this theme.",
                        "Roger Dodger is at least compelling as a variation on the theme.",
                        "Roger Dodger is one of the least compelling variations on this theme.",
                        "Not such a badass after all.",  # Capitalized negation with slang
                        "Without a doubt, an excellent idea."  # "without {any} doubt" as negation
                        ]
    print("----------------------------------------------------")
    print(" - Analyze examples of tricky sentences that cause trouble to other sentiment analysis tools.")
    print("  -- special case idioms - e.g., 'never good' vs 'never this good', or 'bad' vs 'bad ass'.")
    print("  -- special uses of 'least' as negation versus comparison \n")
    for sentence in tricky_sentences:
        vs = analyzer.polarity_scores(sentence)
        print("{:-<69} {}".format(sentence, str(vs)))
    print("----------------------------------------------------")

    # input("\nPress Enter to continue the demo...\n")  # for DEMO purposes...

    print("----------------------------------------------------")
    print(
        " - VADER works best when analysis is done at the sentence level (but it can work on single words or entire novels).")
    paragraph = "It was one of the worst movies I've seen, despite good reviews. Unbelievably bad acting!! Poor direction. VERY poor production. The movie was bad. Very bad movie. VERY BAD movie!"
    print("  -- For example, given the following paragraph text from a hypothetical movie review:\n\t'{}'".format(
        paragraph))
    print(
        "  -- You could use NLTK to break the paragraph into sentence tokens for VADER, then average the results for the paragraph like this: \n")
    # simple example to tokenize paragraph into sentences for VADER
    from nltk import tokenize

    sentence_list = tokenize.sent_tokenize(paragraph)
    paragraphSentiments = 0.0
    for sentence in sentence_list:
        vs = analyzer.polarity_scores(sentence)
        print("{:-<69} {}".format(sentence, str(vs["compound"])))
        paragraphSentiments += vs["compound"]
    print("AVERAGE SENTIMENT FOR PARAGRAPH: \t" + str(round(paragraphSentiments / len(sentence_list), 4)))
    print("----------------------------------------------------")

    # input("\nPress Enter to continue the demo...\n")  # for DEMO purposes...

    print("----------------------------------------------------")
    print(" - Analyze sentiment of IMAGES/VIDEO data based on annotation 'tags' or image labels. \n")
    conceptList = ["balloons", "cake", "candles", "happy birthday", "friends", "laughing", "smiling", "party"]
    conceptSentiments = 0.0
    for concept in conceptList:
        vs = analyzer.polarity_scores(concept)
        print("{:-<15} {}".format(concept, str(vs['compound'])))
        conceptSentiments += vs["compound"]
    print("AVERAGE SENTIMENT OF TAGS/LABELS: \t" + str(round(conceptSentiments / len(conceptList), 4)))
    print("\t")
    conceptList = ["riot", "fire", "fight", "blood", "mob", "war", "police", "tear gas"]
    conceptSentiments = 0.0
    for concept in conceptList:
        vs = analyzer.polarity_scores(concept)
        print("{:-<15} {}".format(concept, str(vs['compound'])))
        conceptSentiments += vs["compound"]
    print("AVERAGE SENTIMENT OF TAGS/LABELS: \t" + str(round(conceptSentiments / len(conceptList), 4)))
    print("----------------------------------------------------")

    # input("\nPress Enter to continue the demo...")  # for DEMO purposes...

    do_translate = input(
        "\nWould you like to run VADER demo examples with NON-ENGLISH text? \n (Note: requires Internet access and uses the 'requests' library) \n Type 'y' or 'n', then press Enter: ")
    if do_translate.lower().lstrip().__contains__("y"):
        import requests
        print("\n----------------------------------------------------")
        print(" - Analyze sentiment of NON ENGLISH text...for example:")
        print("  -- French, German, Spanish, Italian, Russian, Japanese, Arabic, Chinese(Simplified) , Chinese(Traditional)")
        print("  -- many other languages supported. \n")
        languages = ["English", "French", "German", "Spanish", "Italian", "Russian", "Japanese", "Arabic", "Chinese(Simplified)", "Chinese(Traditional)"]
        language_codes = ["en", "fr", "de", "es", "it", "ru", "ja", "ar", "zh-CN", "zh-TW"]
        nonEnglish_sentences = ["I'm surprised to see just how amazingly helpful VADER is!",
                                "Je suis surpris de voir comment VADER est incroyablement utile !",
                                "Ich bin überrascht zu sehen, nur wie erstaunlich nützlich VADER!",
                                "Me sorprende ver sólo cómo increíblemente útil VADER!",
                                "Sono sorpreso di vedere solo come incredibilmente utile VADER è!",
                                "Я удивлен увидеть, как раз как удивительно полезно ВЕЙДЕРА!",
                                "私はちょうどどのように驚くほど役に立つベイダーを見て驚いています!",
                                "أنا مندهش لرؤية فقط كيف مثير للدهشة فيدر فائدة!",
                                "我很惊讶地看到VADER是如此有用!",
                                "我很驚訝地看到VADER是如此有用!"
                                ]
        for sentence in nonEnglish_sentences:
            to_lang = "en"
            from_lang = language_codes[nonEnglish_sentences.index(sentence)]
            if (from_lang == "en") or (from_lang == "en-US"):
                translation = sentence
                translator_name = "No translation needed"
            else:  # please note usage limits for My Memory Translation Service:   http://mymemory.translated.net/doc/usagelimits.php
                # using   MY MEMORY NET   http://mymemory.translated.net
                api_url = "http://mymemory.translated.net/api/get?q={}&langpair={}|{}".format(sentence, from_lang,
                                                                                              to_lang)
                hdrs = {
                    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
                    'Accept-Encoding': 'none',
                    'Accept-Language': 'en-US,en;q=0.8',
                    'Connection': 'keep-alive'}
                response = requests.get(api_url, headers=hdrs)
                response_json = json.loads(response.text)
                translation = response_json["responseData"]["translatedText"]
                translator_name = "MemoryNet Translation Service"
            vs = analyzer.polarity_scores(translation)
            print("- {: <8}: {: <69}\t {} ({})".format(languages[nonEnglish_sentences.index(sentence)], sentence,
                                                       str(vs['compound']), translator_name))
        print("----------------------------------------------------")

    print("\n\n Demo Done!")
