from .base_parameter_helper import BaseParameterHelper

class LoggerParameterHelper(BaseParameterHelper):
    """
    A helper to generate the parameter templates for logger.

    Template:

        ```
        {
            "sender": "this is relevant for remote job submission scenario",
            "recipient": loggerRecipient,
            "logging_frequency": loggerFreq,
            "logging_path": loggerPath,
            "resultdir": loggerDir,
            "job_name": "Reinforcement learning",
            "job_id": "relevant for remote logging"
        }
        ```

    Usage:
        ```
        # Connect to HandB Ops
        loggerRecipient = "neptune"
        loggerFreq = 20
        loggerPath = "xtalpi/REINVENT"
        loggerDir = os.path.abspath("./output")
        jobName = "Reinforcement Learning"

        loggerParameterHelper = LoggerParameterHelper(
            recipient, frequency, logdir, 
            outdir, jobname, jobid
        )
        loggerParameterHelper.generate_template()
        ```

    """
    JSON_TEMPLATE = {
        "sender": "this is relevant for remote job submission scenario",
        "recipient": "local",
        "logging_frequency": 20,
        "logging_path": "xtalpi/REINVENT",
        "resultdir": "./output",
        "job_name": "Reinforcement learning",
        "job_id": "relevant for remote logging"
    }

    def __init__(self, recipient, frequency, logpath, retdir, jobname, jobid):
        self.recipient = recipient
        self.frequency = frequency
        self.logpath = logpath
        self.retdir = retdir
        self.jobname = jobname
        self.jobid = jobid

    def generate_template(self):
        self.JSON_TEMPLATE['recipient'] = self.recipient
        self.JSON_TEMPLATE['logging_frequency'] = self.frequency
        self.JSON_TEMPLATE['logging_path'] = self.logpath
        self.JSON_TEMPLATE['resultdir'] = self.retdir
        self.JSON_TEMPLATE['job_name'] = self.jobname
        self.JSON_TEMPLATE['job_id'] = self.jobid
        return self.JSON_TEMPLATE