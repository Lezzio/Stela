import redis
import atexit

'''
Le système fonctionne avec deux channels.
Le premier permet aux utilisateurs de publier leurs requêtes et au serveur de
les recevoir tandis que le deuxième permet aux utilisateurs de récupérer leurs
réponses que le serveur émettra.
1er : query.$ip
2er : answer.$ip
'''

# Strong Password used
pwd = 'nYn7AyqDcHxgYWUvLJxpJvRQxOeMVK9ah1wkNZrGrLIsh1oeXfcOkejb2zL0FMUDZcGWGCcBJaTV1CHChuwPtC0FftG8ieKSc3gLAfuim9BJxBldprLv1qeMTg7p48cL'


class RedisManager:
    """
    Class which manages redis based server for stela
    """

    def __init__(self, function):
        self.r = None
        self.function = function
        self.r = redis.Redis(host='149.202.55.170', port='6379', db=0, password='foobared')
        pub = self.r.pubsub()
        pub.psubscribe(**{'query': self.message_handler})
        self.thread = pub.run_in_thread(sleep_time=0.01)
        atexit.register(self.close)  # Avoid daemon thread on exit

    def message_handler(self, msg):
        """
        Handle the messages recieved by users
        :param msg: Redis message object
        :return: None
        """
        target, content = str(msg['data'].decode('utf-8')).split("&:")
        replyback_channel = 'answer.' + target
        self.r.publish(replyback_channel, self.function(content))
        print(target + ' asked ' + content)

    def close(self):
        print('Destroying thread...')
        self.thread.stop()
