import tensorflow as tf

# fetch
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
intermed = tf.add(input2, input3)
mul = tf.multiply(input1, intermed)

with tf.Session() as sess:
  result = sess.run([mul, intermed])
  print (result)

# feed
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

with tf.Session() as sess:
  print( sess.run([output], feed_dict={input1:[7.], input2:[2.]}) )

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)

print(node1, node2)
with tf.Session() as sess:
    print(sess.run([node1, node2]))
    node3 = tf.add(node1, node2)
    node4 = node1 + node2
    print(node3)
    print(sess.run(node3))
    print(node4)
    print(sess.run(node4))

sess = tf.Session()
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b
print(sess.run(adder_node, {a:2, b:3}))
print(sess.run(adder_node, {a:[1, 3], b:[3, 4]}))

add_and_triple = adder_node * 3
print(sess.run(add_and_triple, {a: 3, b: 4}))

W = tf.Variable([-1.], dtype=tf.float32)
b = tf.Variable([1.], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()

sess.run(init)
for i in range(1):
    sess.run(train, {x:[1,2,3,4,5], y:[0, -1, -2, -3, -4]})
print(sess.run([W, b]))