QA_PROMPT = """Answer the question.
Your output MUST in ONE OR FEW WORDS.

Here are some examples:
    # 
    Question: Who is Catherine Of Pomerania, Countess Palatine Of Neumarkt’s father-in-law?
    Answer: Rupert III of the Palatinate
    #
    Question: Which film has the director died first, Crimen A Las Tres or The Working Class Goes To Heaven?
    Answer: The Working Class Goes to Heaven
    #
    Question: Which genus of moth in the world's seventh-largest country contains only one species?
    Answer: Crambidae
    # 
    Question: What is the length of the track where the 2013 Liqui Moly Bathurst 12 Hour was staged?
    Answer: 6.213 km long

Question: {question}
Answer: """
