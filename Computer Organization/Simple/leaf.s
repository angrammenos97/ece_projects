	.text
	.align	2
	.globl	leaf
	.set	nomips16
	.set	nomicromips
	.ent	leaf

leaf:
	addi $sp, $sp, -4
	sw   $s0, 0($sp)
	add  $t0, $a0, $a1
	add  $t1, $a2, $a3
	sub  $s0, $t0, $t1
	add  $v0, $s0, $zero
	lw   $s0, 0($sp)
	addi $sp, $sp, 4
	jr   $ra

main:
	.frame	$fp,56,$31		# vars= 24, regs= 2/0, args= 16, gp= 8
	.mask	0xc0000000,-4
	.fmask	0x00000000,0
	.set	noreorder
	.set	nomacro
	addiu	$sp,$sp,-56
	sw	$31,52($sp)
	sw	$fp,48($sp)
	move	$fp,$sp
	li	$2,2			# 0x2
	sw	$2,28($fp)
	li	$2,3			# 0x3
	sw	$2,32($fp)
	li	$2,4			# 0x4
	sw	$2,36($fp)
	li	$2,5			# 0x5
	sw	$2,40($fp)
	lw	$7,40($fp)
	lw	$6,36($fp)
	lw	$5,32($fp)
	lw	$4,28($fp)
	.option	pic0
	jal	leaf
	nop

	move    $a0,$v0
	li      $v0,1
	syscall

	.option	pic2
	sw	$2,44($fp)
	nop
	move	$sp,$fp
	lw	$31,52($sp)
	lw	$fp,48($sp)
	addiu	$sp,$sp,56
	j	$31
	nop

	.set	macro
	.set	reorder
	.end	main
