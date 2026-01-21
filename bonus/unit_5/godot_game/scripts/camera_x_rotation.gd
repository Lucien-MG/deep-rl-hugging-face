extends Node3D
class_name CameraXRotation

## Allows the camera to rotate around the player along X axis
## without affecting the player's rotation.

@export var camera_x_max_rotation_degrees: float = 45.0

@onready var initial_transform = transform


func rotate_camera_x(x: float, delta: float):
	rotation.x -= x * delta
	rotation_degrees.x = clampf(
		rotation_degrees.x, -camera_x_max_rotation_degrees, camera_x_max_rotation_degrees
	)


func reset():
	transform = initial_transform
