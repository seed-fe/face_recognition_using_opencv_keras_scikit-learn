# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 16:12:59 2018

@author: 123
"""

import logging
import sys

def log(message):
    # 实例化日志对象，参考https://docs.python.org/3/howto/logging.html#logging-basic-tutorial
    logger = logging.getLogger('Test')
    # 指定日志输出格式，-8s表示八位字符串，asctime表示日志记录时间，levelname表示日志等级，message是日志具体信息，formatter object具体参考https://docs.python.org/3/howto/logging.html#formatters
    # 三个参数，fmt是日志信息格式，datefmt是日期时间格式，style是fmt参数的样式
    # 如果不指定fmt，那么就只有原始信息
    # 如果不指定datefmt，就会显示到毫秒
    # 具体时间格式参考https://docs.python.org/3/howto/logging.html#displaying-the-date-time-in-messages和https://docs.python.org/3/library/time.html#time.strftime
    # style indicator默认是'%'， 此时message format string uses %(<dictionary key>)s，下面的asctime、levelname和message都是LogRecord attributes
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s', '%m/%d/%Y %I:%M:%S %p')
    # 控制台日志，StreamHandler是通过sys.stdout发送到console，FileHandler则发送到文件，参考https://docs.python.org/3/library/logging.handlers.html#logging.StreamHandler
    console_handler = logging.StreamHandler(sys.stdout)
    # 给输出到控制台的日志指定格式
    console_handler.formatter = formatter
    # 文件日志
    file_handler = logging.FileHandler("performance.log")
    file_handler.formatter = formatter
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # 指定日志的最低输出级别，默认为WARN级别
    logger.setLevel(logging.INFO)
    logger.info(message)
    logger.removeHandler(file_handler)
    logger.removeHandler(console_handler)