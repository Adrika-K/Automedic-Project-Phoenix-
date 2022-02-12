import turtle
import time
wn=turtle.Screen()
wn.title("Team Project Phoenix")
wn.bgcolor("black")

#Draw a box around the light
pen=turtle.Turtle()
pen.color("yellow")
pen.width(3)
pen.hideturtle()
pen.penup()
pen.goto(-30,60)
pen.pendown()
pen.fd(60)
pen.rt(90)
pen.fd(120)
pen.rt(90)
pen.fd(60)
pen.rt(90)
pen.fd(120)

#Red Light
red_light=turtle.Turtle()
red_light.shape("circle")
red_light.color("grey")
red_light.penup()
red_light.goto(0,40)

#Yellow Light
yellow_light=turtle.Turtle()
yellow_light.shape("circle")
yellow_light.color("grey")
yellow_light.penup()
yellow_light.goto(0,0)

#Green Light
green_light=turtle.Turtle()
green_light.shape("circle")
green_light.color("grey")
green_light.penup()
green_light.goto(0,-40)
red_light.color("red")

while True: #Condition
    time.sleep(1.5)
    red_light.color("grey")
    green_light.color("green")