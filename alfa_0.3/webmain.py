# coding:utf-8

import web
import HVS_EGM_SVM as HD
render = web.template.render('templates/')
urls = (
    '/', 'index'
)

class index:
    def GET(self):
        filename = web.input(filename = "/Users/nevin47/Desktop/Project/Academic/Code/Python/SVM/UnbalancedDataSVM/DataSet/CreditOriginData2.csv")
        # filename = '/Users/nevin47/Desktop/Project/Academic/Code/Python/SVM/UnbalancedDataSVM/DataSet/CreditOriginData2.csv' # 设置读取文件
        scaler = 1 # 决定是否归一化数据
        MAXFEATURENUM = 5
        Gmeas, Fmeansure = HD.main(filename.filename, scaler, MAXFEATURENUM, kernel='rbf', C=5.0, gamma= 1)
        # i = web.input(name=None)
        return render.index([Gmeas, Fmeansure])

if __name__ == "__main__":
    app = web.application(urls, globals())
    app.run()