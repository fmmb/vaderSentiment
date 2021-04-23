====================================
VADER-Sentiment-Analysis (Adaptation for Portuguese)
====================================

Code Examples
------------------------------------
::

	from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
	#note: depending on how you installed (e.g., using source code download versus pip install), you may need to import like this:
	#from vaderSentiment import SentimentIntensityAnalyzer

    # --- examples -------
    sentences = ["VADER is smart, handsome, and funny."]
    analyzer = SentimentIntensityAnalyzer('en')
    for sentence in sentences:
        vs = analyzer.polarity_scores(sentence)
        print("{:-<65} {}".format(sentence, str(vs)))
        
    sentences = ["O VADER é muito fixe, interessante e divertido."]
    analyzer = SentimentIntensityAnalyzer('pt')
    for sentence in sentences:
        vs = analyzer.polarity_scores(sentence)
        print("{:-<65} {}".format(sentence, str(vs)))