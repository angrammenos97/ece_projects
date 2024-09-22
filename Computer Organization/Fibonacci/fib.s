	.text
	.align	2
	.globl	fib
fib:
	addiu	$sp,$sp,-40
	sw	$31,36($sp)
	sw	$fp,32($sp)
	move	$fp,$sp
	sw	$4,40($fp)
	lw	$2,40($fp)
	slt	$2,$2,2
	beq	$2,$0,$L2
	nop

	lw	$2,40($fp)
	b	$L3
	nop

$L2:
	lw	$2,40($fp)
	addiu	$2,$2,-1
	move	$4,$2
	jal	fib
	nop

	sw	$2,24($fp)
	lw	$2,40($fp)
	addiu	$2,$2,-2
	move	$4,$2
	jal	fib
	nop

	sw	$2,28($fp)
	lw	$3,24($fp)
	lw	$2,28($fp)
	addu	$2,$3,$2
$L3:
	move	$sp,$fp
	lw	$31,36($sp)
	lw	$fp,32($sp)
	addiu	$sp,$sp,40
	j	$31
	nop

	.align	2
	.globl	main        
main:
	addiu	$sp,$sp,-40
	sw	$31,36($sp)
	sw	$fp,32($sp)
	move	$fp,$sp
	li	$4,5 			# 0x6
	jal	fib
	nop

	sw	$2,28($fp)
	nop

        move    $a0, $v0        # prepare print
        li      $v0, 1          # integer print

        syscall
        
	move	$sp,$fp
	lw	$31,36($sp)
	lw	$fp,32($sp)
	addiu	$sp,$sp,40
	j	$31
	nop
