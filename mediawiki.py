import wikipedia
from SPARQLWrapper import SPARQLWrapper, JSON

wikipedia.set_lang('fr')
#print(wikipedia.summary('Brad Pitt'))
#print(wikipedia.geosearch(48.866667,2.333333))
BP = wikipedia.WikipediaPage('Coluche')
#print(wikipedia.summary(wikipedia.random(pages=1)))

def isURL(string):
    return str(string).startswith('http://')

def getValue(person, value, n):

    stopwords = ['quoi', 'quand', 'qui', 'où', 'et', 'est', 'il', 'ils', 'comment', 'pourquoi', 'combien', 'je', 'tu',
                 'nous', 'vous', 'elles', 'elle', 'on', 'ont', 'donc', 'or', 'ni', 'car', '-']
    querywords = person.split()
    resultwords = [word for word in querywords if word.lower() not in stopwords]
    person = ' '.join(resultwords)
    personList = wikipedia.search(person)
    person = personList[n]
    person = person.replace(' ', '_')
    response = ''
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setQuery("""
    PREFIX dbpedia-owl: <http://dbpedia.org/ontology/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    SELECT ?birth ?label
    WHERE
    {
      <http://dbpedia.org/resource/"""+person+"""> dbpedia-owl:""" + value + """ ?birth ;
      rdfs:label ?label .
      FILTER (lang(?label) = 'fr')
    }
    """)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    for result in results["results"]["bindings"]:
        response = result["birth"]["value"]

    if response == '' and n+1 <= len(personList):
        response = getValue(person, value, n+1)
    elif response == '' and n > len(personList):
        response = 'Désolé j\ai pas pu trouver une réponse à votre requête'
    elif isURL(response):
        sparql2 = SPARQLWrapper("http://dbpedia.org/sparql")
        sparql2.setQuery("""
            PREFIX dbpedia-owl: <http://dbpedia.org/ontology/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            SELECT ?birth
            WHERE
            {
              <""" + response + """> rdfs:label ?birth .
              FILTER (lang(?birth) = 'fr')
            }
            """)
        sparql2.setReturnFormat(JSON)
        resultats = sparql2.query().convert()

        for resulte in resultats["results"]["bindings"]:
            response = resulte["birth"]["value"]

    return response

'''
print(getValue('Tesla', 'deathDate',0))
print(getValue('Tesla', 'birthPlace',0))
print(getValue('Tesla', 'citizenship',0))
print(getValue('Tesla', 'deathCause',0))
print(getValue('Tesla', 'deathPlace',0))
print(getValue('Tesla', 'education',0))
print(getValue('Tesla', 'ethnicity',0))
'''


from yandex_translate import YandexTranslate

def gettranslation(text, target):
    translate = YandexTranslate('trnsl.1.1.20170410T163606Z.0ef0c73fe6d1d9af.d79b976abf7f8b60699b87a4af79c6784132c8f3')
    langdetected = translate.detect(text)
    trad = translate.translate(text, langdetected+'-'+target)
    return trad

#print(gettranslation('Привет, мир!', 'en')['text'][0])
#print(gettranslation('Bonjour, je suis Geoffrey!', 'de')['text'][0])

#33 Languages: {'sq', 'hy', 'tr', 'sr', 'hu', 'pt', 'it', 'mk', 'no', 'de', 'pl', 'es', 'hr', 'ru', 'lt', 'ro', 'fi', 'et', 'en', 'sl', 'uk', 'el', 'ca', 'fr', 'nl', 'lv', 'sk', 'bg', 'cs', 'az', 'sv', 'da', 'be'}