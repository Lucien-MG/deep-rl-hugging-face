extends Area3D
class_name Chest

## Treasure chest, can be opened by the player if the player has a key collected,
## resulting in game victory.

@onready var chest_locked := $chest_closed
@onready var chest_open := $chest_open
@onready var particles := $GPUParticles3D

var _locked: bool = true
var _opened: bool
var _player_in_area: Player

signal opened


func _physics_process(_delta):
	if _player_in_area and _player_in_area.use_action_requested and not _locked:
		open()


func close():
	chest_open.visible = false
	chest_locked.visible = true
	particles.emitting = false


func open():
	if not _opened:
		chest_open.visible = true
		chest_locked.visible = false
		_opened = true
		opened.emit()
		particles.emitting = true

## This signal is connected to the Key
func _on_key_collected():
	_locked = false


func _on_body_entered(body):
	if body is Player:
		_player_in_area = body


func _on_body_exited(body):
	if body is Player:
		_player_in_area = null


func reset():
	close()
	_locked = true
	_opened = false
	_player_in_area = null
