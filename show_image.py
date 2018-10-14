# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 10:50:53 2018

@author: 123
"""

import sys
from PyQt5 import QtGui, QtWidgets

def show_image(image_path='spyder.png'):
    app = QtWidgets.QApplication(sys.argv)
    pixmap = QtGui.QPixmap(image_path)
    screen = QtWidgets.QLabel() # The QLabel widget provides a text or image display. 
    screen.setPixmap(pixmap)
    screen.showFullScreen()
    # sys.exit() 会抛出一个异常: SystemExit，如果这个异常没有被捕获，那么python解释器将会退出。如果有捕获该异常的代码，那么这些代码还是会执行。
    sys.exit(app.exec_())


if __name__ == '__main__':
    show_image()