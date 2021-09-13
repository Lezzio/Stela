import wikipedia
import mediawiki

def wiki(content, property):
    return mediawiki.getValue(content, property, 0)