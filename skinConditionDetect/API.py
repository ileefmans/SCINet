from flask import Flask
from flask_restful import Api, Resource

app = Flask(__name__)
api = Api(app)

names = {"ian":{"age":26, "gender":'male'},
		"bill": {"age":33, "gender":'male'}}


class ClassName(Resource):
	def get(self, name, ):
		return names[name]

	

api.add_resource(ClassName, "/classname/<string:name>")



















if __name__ == "__main__":
	app.run(debug=True)





