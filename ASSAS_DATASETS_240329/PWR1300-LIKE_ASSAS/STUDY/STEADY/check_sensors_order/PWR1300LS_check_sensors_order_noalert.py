#!/usr/bin/env python3

from CheckSensorsOrder import CheckSensorsOrder
import os

# Launch computation
computation = CheckSensorsOrder(os.path.join("..","PWR1300-STEADY_LIKE_SIMPLIFIED_ASSAS.mdat"))
computation.run()
