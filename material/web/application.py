from sklearn.externals import joblib
from flask import Flask
from flask import render_template
from flask import request
app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')
    
@app.route('/predict')
def predict(predict=None):
    T = request.args.get('T', 0)
    MPa = request.args.get('MPa', 0)
    J = request.args.get('J', 0)
    clf = joblib.load('./GBDT.sav')
    try:
        predict = int(clf.predict([[T,MPa,J]])[0])
    except BaseException:
        predict = '预测失败，请检查数据！'
    return render_template('index.html', predict=predict,T=T,MPa=MPa,J=J)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)# 给予外部访问权限