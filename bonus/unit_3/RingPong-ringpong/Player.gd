extends Node3D

@export var rotation_speed = 3.0
@onready var ball = get_node("../Ball")
var needs_reset = false

func game_over():
	needs_reset = true

func _physics_process(delta):
	if needs_reset:
		ball.reset()
		needs_reset = false
		return
		
	var movement = Input.get_axis("rotate_anticlockwise", "rotate_clockwise")
	rotate_y(movement*delta*rotation_speed)


func _on_area_3d_body_entered(body):
	print("ball hit paddle")
