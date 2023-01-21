from flask import Flask, request, jsonify, render_template
import numpy as np
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index4.html')

@app.route("/process_image", methods=["POST"])
def process_image():
    data = request.get_json()
    list = data["imageArray"]
    # process the image array here
    array = np.array(list)
    print(array.shape)
    return jsonify({"message":"Image processed successfully"})


if __name__ == '__main__':
   app.run(debug = True)