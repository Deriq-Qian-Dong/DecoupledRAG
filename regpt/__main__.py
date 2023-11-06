# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
本文件允许模块包以python -m regpt方式直接执行。

Authors: dongqian06(dongqian06@baidu.com)
Date:    2023/11/06 19:22:21
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals


import sys
from regpt.cmdline import main
sys.exit(main())
