from flask import Flask, render_template
import random

app = Flask(__name__)

@app.route('/')
def index():
    # Simulated vehicle count for UI
    lane_counts = [random.randint(5, 20), random.randint(3, 15)]
    return render_template("index.html", lane_counts=lane_counts)

if __name__ == '__main__':
    app.run(debug=True)
