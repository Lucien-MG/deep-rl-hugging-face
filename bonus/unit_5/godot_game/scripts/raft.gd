extends AnimatableBody3D
class_name Raft

## Raft that moves in movement_direction specified and inverts direction on hitting an obstacle.

@export var sway_speed := 1.0
@export var sway_strength := 0.1
@export var movement_speed := 1.45
@export var movement_direction := Vector3(0, 0, -1)

var angle := 0.0
var movement_direction_multiplier := 1.0
@onready var initial_rotation: Vector3 = rotation
@onready var initial_position: Vector3 = position


func _physics_process(delta):
	# Adds some rotation to the raft
	angle = fmod(angle + sway_speed * delta, TAU)
	rotation = initial_rotation + Vector3(sin(angle) * sway_strength, cos(angle) * sway_strength, 0)
	# Moves the raft
	position += (movement_direction * movement_direction_multiplier * movement_speed * delta)


func _on_area_3d_body_entered(_body):
	# Flips the movement direction when the raft encounters an obstacle
	movement_direction_multiplier = -movement_direction_multiplier

func reset():
	movement_direction_multiplier = 1.0
	position = initial_position
