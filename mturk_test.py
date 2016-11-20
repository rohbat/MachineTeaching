import boto.mturk.connection
 
sandbox_host = 'mechanicalturk.sandbox.amazonaws.com'
real_host = 'mechanicalturk.amazonaws.com'
 
mturk = boto.mturk.connection.MTurkConnection(
    aws_access_key_id = 'AKIAIGJI3VXYBR765VLA',
    aws_secret_access_key = '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',
    host = sandbox_host,
    debug = 1 # debug = 2 prints out all requests.
)
 
print boto.Version # 2.29.1
print mturk.get_account_balance() # [$10,000.00]