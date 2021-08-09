from flask import Flask, abort
from flask import request, jsonify
from flask_cors import CORS, cross_origin
from flask_restful import Resource, Api

import subprocess
import os.path
from os import path

app = Flask(__name__)
api = Api(app)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'


class startTraining(Resource):
    def __init__(self):
        super(startTraining, self).__init__()

    @cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
    def post(self):
        try:
            args = request.get_json()
            # print(args)

            subprocess.Popen(
                ['python', 'train.py', 
                '--memory', args['memory'],
                '--exploration_rate_decay', args['exploration_rate_decay'],
                '--learn_every', args['learn_every'],
                '--sync_every', args['sync_every'],
                ]
            )
            return jsonify({"data": "Starting training process for Mario ...",
                            "error": None
                            }), 201        
        except KeyError as err:
            return jsonify({"data": None,
                            "error": "Wrong/missing key or missing JSON body"
                            }), 404
        except TypeError as err:
            return jsonify({"data": None,
                            "error": "JSON key values are non-string"
                            }), 400
        except subprocess.CalledProcessError as err:
            return jsonify({"data": None,
                            "error": "CalledProcessError"
                            }), 504


class startTesting(Resource):
    def __init__(self):
        super(startTesting, self).__init__()

    @cross_origin(origin='*', headers=['Content-Type'])
    def post(self):
        try:
            args = request.get_json()
            # print(args)

            if path.exists(args['checkpoint']):
                subprocess.Popen(
                    ['python', 'test.py', 
                    '--checkpoint', args['checkpoint'],
                    ]
                )
                return jsonify({"data": "Starting evaluation for Mario ...",
                                "error": None
                                }), 201
            else:
                return jsonify({"data": None,
                                "error": "Checkpoint path is invalid"
                                }), 405  
        except KeyError as err:
            return jsonify({"data": None,
                            "error": "Wrong/missing key or missing JSON body"
                            }), 404
        except TypeError as err:
            return jsonify({"data": None,
                            "error": "Checkpoint path is non-string"
                            }), 400    
        except subprocess.CalledProcessError as err:
            return jsonify({"data": None,
                            "error": "CalledProcessError"
                            }), 504


class playGame(Resource):
    def __init__(self):
        super(playGame, self).__init__()

    @cross_origin(origin='*', headers=['Content-Type'])
    def post(self):
        try:
            subprocess.Popen(
                ['gym_super_mario_bros', 
                '-e', 'SuperMarioBros-v0',
                '-m', 'human',
                ]
            )
            if AttributeError:
                return jsonify({"data": None,
                                "error": "Bad request: game cannot be played"
                                }), 400
            else:
                return jsonify({"data": "Starting Mario game ...",
                            "error": None
                            }), 201 
        except subprocess.CalledProcessError as err:
            return jsonify({"data": None,
                            "error": "CalledProcessError"
                            }), 504


##############################################################
#                      ENDPOINTS                             #
##############################################################
api.add_resource(startTraining, "/train")
api.add_resource(startTesting, "/test")
api.add_resource(playGame, "/play")

if __name__ == '__main__':

    app.run(host="0.0.0.0", port=5001, debug=True)