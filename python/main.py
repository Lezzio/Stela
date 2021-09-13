#!/usr/bin/python
# vim: set fileencoding=utf-8 :
from redismanager import RedisManager
import keras_stela
import functions

global_answer = ['#wiki&birthDate',
                 '#wiki&deathDate',
                 'Je vais très bien et toi ?', #0
                 'Mon prénom est Stela', #1
                 'Je peux essayer La la la la la',
                 'Mes créateurs sont Allane, Paul et Julien',
                 'J\'ai été crée en 2017',
                 'Je suis Stela',
                 'J\'adore les mathématiques',
                 'Un indice ça commence par un M',
                 'C\'est toi',
                 'Oui je viens de le régler'
                 ]

def proceedRequest(content):
    id = int(keras_stela.predict(content))
    print(id)
    if id < len(global_answer) and global_answer[id] is not None:
        answer = global_answer[id]
        print('-> ', answer)
        if answer.startswith('#'):
            preanswer = answer[1:].split('&')
            print(preanswer)
            method = getattr(functions, preanswer[0])
            return method(content, preanswer[1])
        return str(answer)
    else: return 'Je n\'ai pas compris t\'a question'


def main():
    print('Starting...')
    redisManager = RedisManager(proceedRequest)
    print('Ready to receive queries')


if __name__ == '__main__':
    main()
