	.text
	.def	 @feat.00;
	.scl	3;
	.type	0;
	.endef
	.globl	@feat.00
.set @feat.00, 0
	.file	"intersectors.ispc"
	.def	 "operator-___s_5B_vyvec3_5D_s_5B_vyvec3_5D_";
	.scl	2;
	.type	32;
	.endef
	.globl	"operator-___s_5B_vyvec3_5D_s_5B_vyvec3_5D_" # -- Begin function operator-___s_5B_vyvec3_5D_s_5B_vyvec3_5D_
	.p2align	4, 0x90
"operator-___s_5B_vyvec3_5D_s_5B_vyvec3_5D_": # @operator-___s_5B_vyvec3_5D_s_5B_vyvec3_5D_
# %bb.0:                                # %allocas
	vmovaps	(%r8), %ymm2
	vmovaps	(%rdx), %ymm1
	vmovaps	(%rcx), %ymm0
	movq	48(%rsp), %rax
	movq	40(%rsp), %rcx
	vsubps	(%r9), %ymm0, %ymm0
	vsubps	(%rcx), %ymm1, %ymm1
	vsubps	(%rax), %ymm2, %ymm2
	retq
                                        # -- End function
	.def	 SphereHitN___un_3C_unf_3E_un_3C_unf_3E_un_3C_unf_3E_un_3C_unf_3E_REFs_5B_unvec3_5D_REFs_5B_unvec3_5D_un_3C_unf_3E_uni;
	.scl	2;
	.type	32;
	.endef
	.globl	__real@80000000                 # -- Begin function SphereHitN___un_3C_unf_3E_un_3C_unf_3E_un_3C_unf_3E_un_3C_unf_3E_REFs_5B_unvec3_5D_REFs_5B_unvec3_5D_un_3C_unf_3E_uni
	.section	.rdata,"dr",discard,__real@80000000
	.p2align	2
__real@80000000:
	.long	0x80000000                      # float -0
	.globl	__real@4b189680
	.section	.rdata,"dr",discard,__real@4b189680
	.p2align	2
__real@4b189680:
	.long	0x4b189680                      # float 1.0E+7
	.globl	__ymm@0000000700000006000000050000000400000003000000020000000100000000
	.section	.rdata,"dr",discard,__ymm@0000000700000006000000050000000400000003000000020000000100000000
	.p2align	5
__ymm@0000000700000006000000050000000400000003000000020000000100000000:
	.long	0                               # 0x0
	.long	1                               # 0x1
	.long	2                               # 0x2
	.long	3                               # 0x3
	.long	4                               # 0x4
	.long	5                               # 0x5
	.long	6                               # 0x6
	.long	7                               # 0x7
	.text
	.globl	SphereHitN___un_3C_unf_3E_un_3C_unf_3E_un_3C_unf_3E_un_3C_unf_3E_REFs_5B_unvec3_5D_REFs_5B_unvec3_5D_un_3C_unf_3E_uni
	.p2align	4, 0x90
SphereHitN___un_3C_unf_3E_un_3C_unf_3E_un_3C_unf_3E_un_3C_unf_3E_REFs_5B_unvec3_5D_REFs_5B_unvec3_5D_un_3C_unf_3E_uni: # @SphereHitN___un_3C_unf_3E_un_3C_unf_3E_un_3C_unf_3E_un_3C_unf_3E_REFs_5B_unvec3_5D_REFs_5B_unvec3_5D_un_3C_unf_3E_uni
# %bb.0:                                # %allocas
	pushq	%r14
	pushq	%rsi
	pushq	%rdi
	pushq	%rbp
	pushq	%rbx
	subq	$240, %rsp
	vmovaps	%xmm15, 224(%rsp)               # 16-byte Spill
	vmovdqa	%xmm14, 208(%rsp)               # 16-byte Spill
	vmovaps	%xmm13, 192(%rsp)               # 16-byte Spill
	vmovaps	%xmm12, 176(%rsp)               # 16-byte Spill
	vmovaps	%xmm11, 160(%rsp)               # 16-byte Spill
	vmovaps	%xmm10, 144(%rsp)               # 16-byte Spill
	vmovaps	%xmm9, 128(%rsp)                # 16-byte Spill
	vmovaps	%xmm8, 112(%rsp)                # 16-byte Spill
	vmovaps	%xmm7, 96(%rsp)                 # 16-byte Spill
	vmovaps	%xmm6, 80(%rsp)                 # 16-byte Spill
	movl	344(%rsp), %r11d
	movq	336(%rsp), %r14
	movq	320(%rsp), %r10
	movq	328(%rsp), %rax
	vmovss	(%rax), %xmm11                  # xmm11 = mem[0],zero,zero,zero
	vmovss	4(%rax), %xmm2                  # xmm2 = mem[0],zero,zero,zero
	vmovss	8(%rax), %xmm1                  # xmm1 = mem[0],zero,zero,zero
	vmulss	%xmm11, %xmm11, %xmm0
	vfmadd231ss	%xmm2, %xmm2, %xmm0     # xmm0 = (xmm2 * xmm2) + xmm0
	vfmadd231ss	%xmm1, %xmm1, %xmm0     # xmm0 = (xmm1 * xmm1) + xmm0
	vbroadcastss	%xmm0, %ymm0
	leal	7(%r11), %esi
	testl	%r11d, %r11d
	cmovnsl	%r11d, %esi
	andl	$-8, %esi
	testl	%esi, %esi
	vmovups	%ymm1, 32(%rsp)                 # 32-byte Spill
	vmovups	%ymm2, (%rsp)                   # 32-byte Spill
	jle	.LBB1_1
# %bb.4:                                # %foreach_full_body.lr.ph
	vbroadcastss	(%r10), %ymm4
	vbroadcastss	4(%r10), %ymm5
	vbroadcastss	8(%r10), %ymm6
	vbroadcastss	%xmm11, %ymm7
	vbroadcastss	%xmm2, %ymm8
	vbroadcastss	%xmm1, %ymm9
	xorl	%ebx, %ebx
	vxorps	%xmm10, %xmm10, %xmm10
	vpcmpeqd	%xmm14, %xmm14, %xmm14
	xorl	%edi, %edi
	jmp	.LBB1_5
	.p2align	4, 0x90
.LBB1_7:                                # %if_done
                                        #   in Loop: Header=BB1_5 Depth=1
	addl	$8, %edi
	addl	$32, %ebx
	cmpl	%esi, %edi
	jge	.LBB1_2
.LBB1_5:                                # %foreach_full_body
                                        # =>This Inner Loop Header: Depth=1
	movslq	%ebx, %rax
	vsubps	(%rcx,%rax), %ymm4, %ymm12
	vsubps	(%rdx,%rax), %ymm5, %ymm13
	vsubps	(%r8,%rax), %ymm6, %ymm15
	vmulps	%ymm8, %ymm13, %ymm2
	vfmadd231ps	%ymm7, %ymm12, %ymm2    # ymm2 = (ymm12 * ymm7) + ymm2
	vfmadd231ps	%ymm9, %ymm15, %ymm2    # ymm2 = (ymm15 * ymm9) + ymm2
	vmulps	%ymm13, %ymm13, %ymm13
	vfmadd231ps	%ymm12, %ymm12, %ymm13  # ymm13 = (ymm12 * ymm12) + ymm13
	vfmadd231ps	%ymm15, %ymm15, %ymm13  # ymm13 = (ymm15 * ymm15) + ymm13
	vmovups	(%r9,%rax), %ymm12
	vfnmadd213ps	%ymm13, %ymm12, %ymm12  # ymm12 = -(ymm12 * ymm12) + ymm13
	vmulps	%ymm0, %ymm12, %ymm12
	vfmsub231ps	%ymm2, %ymm2, %ymm12    # ymm12 = (ymm2 * ymm2) - ymm12
	vcmpltps	%ymm12, %ymm10, %ymm15
	vmovmskps	%ymm15, %ebp
	testb	%bpl, %bpl
	je	.LBB1_6
# %bb.8:                                # %safe_if_run_true
                                        #   in Loop: Header=BB1_5 Depth=1
	vbroadcastss	__real@80000000(%rip), %ymm3 # ymm3 = [-0.0E+0,-0.0E+0,-0.0E+0,-0.0E+0,-0.0E+0,-0.0E+0,-0.0E+0,-0.0E+0]
	vxorps	%ymm3, %ymm2, %ymm2
	vsqrtps	%ymm12, %ymm3
	vsubps	%ymm3, %ymm2, %ymm2
	vdivps	%ymm0, %ymm2, %ymm13
	vcmpltps	%ymm13, %ymm10, %ymm2
	vandps	%ymm2, %ymm15, %ymm2
	vextractf128	$1, %ymm2, %xmm3
	vpackssdw	%xmm3, %xmm2, %xmm2
	vpxor	%xmm2, %xmm14, %xmm3
	vpacksswb	%xmm3, %xmm3, %xmm3
	vpmovmskb	%xmm3, %ebp
	cmpb	$-1, %bpl
	je	.LBB1_10
# %bb.9:                                # %eval_1
                                        #   in Loop: Header=BB1_5 Depth=1
	vpmovsxwd	%xmm2, %ymm3
	vmaskmovps	(%r14,%rax), %ymm3, %ymm3
	vcmpltps	%ymm3, %ymm13, %ymm3
	vextractf128	$1, %ymm3, %xmm1
	vpackssdw	%xmm1, %xmm3, %xmm1
	vpand	%xmm2, %xmm1, %xmm2
.LBB1_10:                               # %logical_op_done
                                        #   in Loop: Header=BB1_5 Depth=1
	vpsllw	$15, %xmm2, %xmm1
	vpmovmskb	%xmm1, %ebp
	testl	$43690, %ebp                    # imm = 0xAAAA
	je	.LBB1_6
# %bb.11:                               # %safe_if_run_true151
                                        #   in Loop: Header=BB1_5 Depth=1
	vpmovzxwd	%xmm2, %ymm1            # ymm1 = xmm2[0],zero,xmm2[1],zero,xmm2[2],zero,xmm2[3],zero,xmm2[4],zero,xmm2[5],zero,xmm2[6],zero,xmm2[7],zero
	vpslld	$31, %ymm1, %ymm1
	vmaskmovps	%ymm13, %ymm1, (%r14,%rax)
.LBB1_6:                                # %safe_if_after_true
                                        #   in Loop: Header=BB1_5 Depth=1
	vcmpnltps	%ymm12, %ymm10, %ymm2
	vextractf128	$1, %ymm2, %xmm3
	vpackssdw	%xmm3, %xmm2, %xmm2
	vpmovmskb	%xmm2, %ebp
	testw	%bp, %bp
	je	.LBB1_7
# %bb.12:                               # %safe_if_run_false175
                                        #   in Loop: Header=BB1_5 Depth=1
	vpmovsxwd	%xmm2, %ymm1
	vbroadcastss	__real@4b189680(%rip), %ymm2 # ymm2 = [1.0E+7,1.0E+7,1.0E+7,1.0E+7,1.0E+7,1.0E+7,1.0E+7,1.0E+7]
	vmaskmovps	%ymm2, %ymm1, (%r14,%rax)
	jmp	.LBB1_7
.LBB1_1:
	xorl	%edi, %edi
.LBB1_2:                                # %partial_inner_all_outer
	cmpl	%r11d, %edi
	jge	.LBB1_3
# %bb.13:                               # %partial_inner_only
	vmovd	%edi, %xmm1
	vpbroadcastd	%xmm1, %ymm1
	vpor	__ymm@0000000700000006000000050000000400000003000000020000000100000000(%rip), %ymm1, %ymm1
	vmovd	%r11d, %xmm2
	vpbroadcastd	%xmm2, %ymm2
	vpcmpgtd	%ymm1, %ymm2, %ymm1
	shll	$2, %edi
	movslq	%edi, %rbx
	vmaskmovps	(%rcx,%rbx), %ymm1, %ymm2
	vxorps	%xmm4, %xmm4, %xmm4
	vmaskmovps	(%rdx,%rbx), %ymm1, %ymm3
	vmaskmovps	(%r8,%rbx), %ymm1, %ymm5
	vbroadcastss	(%r10), %ymm6
	vsubps	%ymm2, %ymm6, %ymm2
	vbroadcastss	4(%r10), %ymm6
	vsubps	%ymm3, %ymm6, %ymm6
	vbroadcastss	8(%r10), %ymm3
	vsubps	%ymm5, %ymm3, %ymm5
	vbroadcastss	%xmm11, %ymm7
	vbroadcastss	(%rsp), %ymm3           # 16-byte Folded Reload
	vbroadcastss	32(%rsp), %ymm8         # 16-byte Folded Reload
	vmulps	%ymm3, %ymm6, %ymm3
	vfmadd231ps	%ymm7, %ymm2, %ymm3     # ymm3 = (ymm2 * ymm7) + ymm3
	vfmadd231ps	%ymm8, %ymm5, %ymm3     # ymm3 = (ymm5 * ymm8) + ymm3
	vmulps	%ymm6, %ymm6, %ymm6
	vfmadd231ps	%ymm2, %ymm2, %ymm6     # ymm6 = (ymm2 * ymm2) + ymm6
	vmaskmovps	(%r9,%rbx), %ymm1, %ymm2
	vfmadd231ps	%ymm5, %ymm5, %ymm6     # ymm6 = (ymm5 * ymm5) + ymm6
	vfnmadd213ps	%ymm6, %ymm2, %ymm2     # ymm2 = -(ymm2 * ymm2) + ymm6
	vmulps	%ymm2, %ymm0, %ymm2
	vfmsub231ps	%ymm3, %ymm3, %ymm2     # ymm2 = (ymm3 * ymm3) - ymm2
	vcmpltps	%ymm2, %ymm4, %ymm5
	vandps	%ymm1, %ymm5, %ymm5
	vmovmskps	%ymm5, %eax
	testb	%al, %al
	je	.LBB1_14
# %bb.16:                               # %safe_if_run_true304
	vextractf128	$1, %ymm5, %xmm6
	vpackssdw	%xmm6, %xmm5, %xmm5
	vbroadcastss	__real@80000000(%rip), %ymm6 # ymm6 = [-0.0E+0,-0.0E+0,-0.0E+0,-0.0E+0,-0.0E+0,-0.0E+0,-0.0E+0,-0.0E+0]
	vxorps	%ymm6, %ymm3, %ymm3
	vsqrtps	%ymm2, %ymm6
	vsubps	%ymm6, %ymm3, %ymm3
	vdivps	%ymm0, %ymm3, %ymm0
	vcmpltps	%ymm0, %ymm4, %ymm3
	vextractf128	$1, %ymm3, %xmm6
	vpackssdw	%xmm6, %xmm3, %xmm3
	vpand	%xmm5, %xmm3, %xmm3
	vpcmpeqd	%xmm5, %xmm5, %xmm5
	vpxor	%xmm5, %xmm3, %xmm5
	vpacksswb	%xmm5, %xmm5, %xmm5
	vpmovmskb	%xmm5, %eax
	cmpb	$-1, %al
	je	.LBB1_18
# %bb.17:                               # %eval_1319
	vpmovsxwd	%xmm3, %ymm5
	vmaskmovps	(%r14,%rbx), %ymm5, %ymm5
	vcmpltps	%ymm5, %ymm0, %ymm5
	vextractf128	$1, %ymm5, %xmm6
	vpackssdw	%xmm6, %xmm5, %xmm5
	vpand	%xmm3, %xmm5, %xmm3
.LBB1_18:                               # %logical_op_done320
	vpsllw	$15, %xmm3, %xmm5
	vpmovmskb	%xmm5, %eax
	testl	$43690, %eax                    # imm = 0xAAAA
	je	.LBB1_14
# %bb.19:                               # %safe_if_run_true348
	vpmovzxwd	%xmm3, %ymm3            # ymm3 = xmm3[0],zero,xmm3[1],zero,xmm3[2],zero,xmm3[3],zero,xmm3[4],zero,xmm3[5],zero,xmm3[6],zero,xmm3[7],zero
	vpslld	$31, %ymm3, %ymm3
	vmaskmovps	%ymm0, %ymm3, (%r14,%rbx)
.LBB1_14:                               # %safe_if_after_true303
	vcmpnltps	%ymm2, %ymm4, %ymm0
	vandps	%ymm0, %ymm1, %ymm0
	vextractf128	$1, %ymm0, %xmm1
	vpackssdw	%xmm1, %xmm0, %xmm0
	vpmovmskb	%xmm0, %eax
	testw	%ax, %ax
	je	.LBB1_3
# %bb.15:                               # %safe_if_run_false380
	vpmovsxwd	%xmm0, %ymm0
	vbroadcastss	__real@4b189680(%rip), %ymm1 # ymm1 = [1.0E+7,1.0E+7,1.0E+7,1.0E+7,1.0E+7,1.0E+7,1.0E+7,1.0E+7]
	vmaskmovps	%ymm1, %ymm0, (%r14,%rbx)
.LBB1_3:                                # %foreach_reset
	vmovaps	80(%rsp), %xmm6                 # 16-byte Reload
	vmovaps	96(%rsp), %xmm7                 # 16-byte Reload
	vmovaps	112(%rsp), %xmm8                # 16-byte Reload
	vmovaps	128(%rsp), %xmm9                # 16-byte Reload
	vmovaps	144(%rsp), %xmm10               # 16-byte Reload
	vmovaps	160(%rsp), %xmm11               # 16-byte Reload
	vmovaps	176(%rsp), %xmm12               # 16-byte Reload
	vmovaps	192(%rsp), %xmm13               # 16-byte Reload
	vmovaps	208(%rsp), %xmm14               # 16-byte Reload
	vmovaps	224(%rsp), %xmm15               # 16-byte Reload
	addq	$240, %rsp
	popq	%rbx
	popq	%rbp
	popq	%rdi
	popq	%rsi
	popq	%r14
	vzeroupper
	retq
                                        # -- End function
	.def	 SphereHitN;
	.scl	2;
	.type	32;
	.endef
	.globl	SphereHitN                      # -- Begin function SphereHitN
	.p2align	4, 0x90
SphereHitN:                             # @SphereHitN
# %bb.0:                                # %allocas
	pushq	%r14
	pushq	%rsi
	pushq	%rdi
	pushq	%rbp
	pushq	%rbx
	subq	$240, %rsp
	vmovaps	%xmm15, 224(%rsp)               # 16-byte Spill
	vmovdqa	%xmm14, 208(%rsp)               # 16-byte Spill
	vmovaps	%xmm13, 192(%rsp)               # 16-byte Spill
	vmovaps	%xmm12, 176(%rsp)               # 16-byte Spill
	vmovaps	%xmm11, 160(%rsp)               # 16-byte Spill
	vmovaps	%xmm10, 144(%rsp)               # 16-byte Spill
	vmovaps	%xmm9, 128(%rsp)                # 16-byte Spill
	vmovaps	%xmm8, 112(%rsp)                # 16-byte Spill
	vmovaps	%xmm7, 96(%rsp)                 # 16-byte Spill
	vmovaps	%xmm6, 80(%rsp)                 # 16-byte Spill
	movl	344(%rsp), %r11d
	movq	336(%rsp), %r14
	movq	320(%rsp), %r10
	movq	328(%rsp), %rax
	vmovss	(%rax), %xmm11                  # xmm11 = mem[0],zero,zero,zero
	vmovss	4(%rax), %xmm2                  # xmm2 = mem[0],zero,zero,zero
	vmovss	8(%rax), %xmm1                  # xmm1 = mem[0],zero,zero,zero
	vmulss	%xmm11, %xmm11, %xmm0
	vfmadd231ss	%xmm2, %xmm2, %xmm0     # xmm0 = (xmm2 * xmm2) + xmm0
	vfmadd231ss	%xmm1, %xmm1, %xmm0     # xmm0 = (xmm1 * xmm1) + xmm0
	vbroadcastss	%xmm0, %ymm0
	leal	7(%r11), %esi
	testl	%r11d, %r11d
	cmovnsl	%r11d, %esi
	andl	$-8, %esi
	testl	%esi, %esi
	vmovups	%ymm1, 32(%rsp)                 # 32-byte Spill
	vmovups	%ymm2, (%rsp)                   # 32-byte Spill
	jle	.LBB2_1
# %bb.4:                                # %foreach_full_body.lr.ph
	vbroadcastss	(%r10), %ymm4
	vbroadcastss	4(%r10), %ymm5
	vbroadcastss	8(%r10), %ymm6
	vbroadcastss	%xmm11, %ymm7
	vbroadcastss	%xmm2, %ymm8
	vbroadcastss	%xmm1, %ymm9
	xorl	%ebx, %ebx
	vxorps	%xmm10, %xmm10, %xmm10
	vpcmpeqd	%xmm14, %xmm14, %xmm14
	xorl	%edi, %edi
	jmp	.LBB2_5
	.p2align	4, 0x90
.LBB2_7:                                # %if_done
                                        #   in Loop: Header=BB2_5 Depth=1
	addl	$8, %edi
	addl	$32, %ebx
	cmpl	%esi, %edi
	jge	.LBB2_2
.LBB2_5:                                # %foreach_full_body
                                        # =>This Inner Loop Header: Depth=1
	movslq	%ebx, %rax
	vsubps	(%rcx,%rax), %ymm4, %ymm12
	vsubps	(%rdx,%rax), %ymm5, %ymm13
	vsubps	(%r8,%rax), %ymm6, %ymm15
	vmulps	%ymm8, %ymm13, %ymm2
	vfmadd231ps	%ymm7, %ymm12, %ymm2    # ymm2 = (ymm12 * ymm7) + ymm2
	vfmadd231ps	%ymm9, %ymm15, %ymm2    # ymm2 = (ymm15 * ymm9) + ymm2
	vmulps	%ymm13, %ymm13, %ymm13
	vfmadd231ps	%ymm12, %ymm12, %ymm13  # ymm13 = (ymm12 * ymm12) + ymm13
	vfmadd231ps	%ymm15, %ymm15, %ymm13  # ymm13 = (ymm15 * ymm15) + ymm13
	vmovups	(%r9,%rax), %ymm12
	vfnmadd213ps	%ymm13, %ymm12, %ymm12  # ymm12 = -(ymm12 * ymm12) + ymm13
	vmulps	%ymm0, %ymm12, %ymm12
	vfmsub231ps	%ymm2, %ymm2, %ymm12    # ymm12 = (ymm2 * ymm2) - ymm12
	vcmpltps	%ymm12, %ymm10, %ymm15
	vmovmskps	%ymm15, %ebp
	testb	%bpl, %bpl
	je	.LBB2_6
# %bb.8:                                # %safe_if_run_true
                                        #   in Loop: Header=BB2_5 Depth=1
	vbroadcastss	__real@80000000(%rip), %ymm3 # ymm3 = [-0.0E+0,-0.0E+0,-0.0E+0,-0.0E+0,-0.0E+0,-0.0E+0,-0.0E+0,-0.0E+0]
	vxorps	%ymm3, %ymm2, %ymm2
	vsqrtps	%ymm12, %ymm3
	vsubps	%ymm3, %ymm2, %ymm2
	vdivps	%ymm0, %ymm2, %ymm13
	vcmpltps	%ymm13, %ymm10, %ymm2
	vandps	%ymm2, %ymm15, %ymm2
	vextractf128	$1, %ymm2, %xmm3
	vpackssdw	%xmm3, %xmm2, %xmm2
	vpxor	%xmm2, %xmm14, %xmm3
	vpacksswb	%xmm3, %xmm3, %xmm3
	vpmovmskb	%xmm3, %ebp
	cmpb	$-1, %bpl
	je	.LBB2_10
# %bb.9:                                # %eval_1
                                        #   in Loop: Header=BB2_5 Depth=1
	vpmovsxwd	%xmm2, %ymm3
	vmaskmovps	(%r14,%rax), %ymm3, %ymm3
	vcmpltps	%ymm3, %ymm13, %ymm3
	vextractf128	$1, %ymm3, %xmm1
	vpackssdw	%xmm1, %xmm3, %xmm1
	vpand	%xmm2, %xmm1, %xmm2
.LBB2_10:                               # %logical_op_done
                                        #   in Loop: Header=BB2_5 Depth=1
	vpsllw	$15, %xmm2, %xmm1
	vpmovmskb	%xmm1, %ebp
	testl	$43690, %ebp                    # imm = 0xAAAA
	je	.LBB2_6
# %bb.11:                               # %safe_if_run_true111
                                        #   in Loop: Header=BB2_5 Depth=1
	vpmovzxwd	%xmm2, %ymm1            # ymm1 = xmm2[0],zero,xmm2[1],zero,xmm2[2],zero,xmm2[3],zero,xmm2[4],zero,xmm2[5],zero,xmm2[6],zero,xmm2[7],zero
	vpslld	$31, %ymm1, %ymm1
	vmaskmovps	%ymm13, %ymm1, (%r14,%rax)
.LBB2_6:                                # %safe_if_after_true
                                        #   in Loop: Header=BB2_5 Depth=1
	vcmpnltps	%ymm12, %ymm10, %ymm2
	vextractf128	$1, %ymm2, %xmm3
	vpackssdw	%xmm3, %xmm2, %xmm2
	vpmovmskb	%xmm2, %ebp
	testw	%bp, %bp
	je	.LBB2_7
# %bb.12:                               # %safe_if_run_false123
                                        #   in Loop: Header=BB2_5 Depth=1
	vpmovsxwd	%xmm2, %ymm1
	vbroadcastss	__real@4b189680(%rip), %ymm2 # ymm2 = [1.0E+7,1.0E+7,1.0E+7,1.0E+7,1.0E+7,1.0E+7,1.0E+7,1.0E+7]
	vmaskmovps	%ymm2, %ymm1, (%r14,%rax)
	jmp	.LBB2_7
.LBB2_1:
	xorl	%edi, %edi
.LBB2_2:                                # %partial_inner_all_outer
	cmpl	%r11d, %edi
	jge	.LBB2_3
# %bb.13:                               # %partial_inner_only
	vmovd	%edi, %xmm1
	vpbroadcastd	%xmm1, %ymm1
	vpor	__ymm@0000000700000006000000050000000400000003000000020000000100000000(%rip), %ymm1, %ymm1
	vmovd	%r11d, %xmm2
	vpbroadcastd	%xmm2, %ymm2
	vpcmpgtd	%ymm1, %ymm2, %ymm1
	shll	$2, %edi
	movslq	%edi, %rbx
	vmaskmovps	(%rcx,%rbx), %ymm1, %ymm2
	vxorps	%xmm4, %xmm4, %xmm4
	vmaskmovps	(%rdx,%rbx), %ymm1, %ymm3
	vmaskmovps	(%r8,%rbx), %ymm1, %ymm5
	vbroadcastss	(%r10), %ymm6
	vsubps	%ymm2, %ymm6, %ymm2
	vbroadcastss	4(%r10), %ymm6
	vsubps	%ymm3, %ymm6, %ymm6
	vbroadcastss	8(%r10), %ymm3
	vsubps	%ymm5, %ymm3, %ymm5
	vbroadcastss	%xmm11, %ymm7
	vbroadcastss	(%rsp), %ymm3           # 16-byte Folded Reload
	vbroadcastss	32(%rsp), %ymm8         # 16-byte Folded Reload
	vmulps	%ymm3, %ymm6, %ymm3
	vfmadd231ps	%ymm7, %ymm2, %ymm3     # ymm3 = (ymm2 * ymm7) + ymm3
	vfmadd231ps	%ymm8, %ymm5, %ymm3     # ymm3 = (ymm5 * ymm8) + ymm3
	vmulps	%ymm6, %ymm6, %ymm6
	vfmadd231ps	%ymm2, %ymm2, %ymm6     # ymm6 = (ymm2 * ymm2) + ymm6
	vmaskmovps	(%r9,%rbx), %ymm1, %ymm2
	vfmadd231ps	%ymm5, %ymm5, %ymm6     # ymm6 = (ymm5 * ymm5) + ymm6
	vfnmadd213ps	%ymm6, %ymm2, %ymm2     # ymm2 = -(ymm2 * ymm2) + ymm6
	vmulps	%ymm2, %ymm0, %ymm2
	vfmsub231ps	%ymm3, %ymm3, %ymm2     # ymm2 = (ymm3 * ymm3) - ymm2
	vcmpltps	%ymm2, %ymm4, %ymm5
	vandps	%ymm1, %ymm5, %ymm5
	vmovmskps	%ymm5, %eax
	testb	%al, %al
	je	.LBB2_14
# %bb.16:                               # %safe_if_run_true223
	vextractf128	$1, %ymm5, %xmm6
	vpackssdw	%xmm6, %xmm5, %xmm5
	vbroadcastss	__real@80000000(%rip), %ymm6 # ymm6 = [-0.0E+0,-0.0E+0,-0.0E+0,-0.0E+0,-0.0E+0,-0.0E+0,-0.0E+0,-0.0E+0]
	vxorps	%ymm6, %ymm3, %ymm3
	vsqrtps	%ymm2, %ymm6
	vsubps	%ymm6, %ymm3, %ymm3
	vdivps	%ymm0, %ymm3, %ymm0
	vcmpltps	%ymm0, %ymm4, %ymm3
	vextractf128	$1, %ymm3, %xmm6
	vpackssdw	%xmm6, %xmm3, %xmm3
	vpand	%xmm5, %xmm3, %xmm3
	vpcmpeqd	%xmm5, %xmm5, %xmm5
	vpxor	%xmm5, %xmm3, %xmm5
	vpacksswb	%xmm5, %xmm5, %xmm5
	vpmovmskb	%xmm5, %eax
	cmpb	$-1, %al
	je	.LBB2_18
# %bb.17:                               # %eval_1234
	vpmovsxwd	%xmm3, %ymm5
	vmaskmovps	(%r14,%rbx), %ymm5, %ymm5
	vcmpltps	%ymm5, %ymm0, %ymm5
	vextractf128	$1, %ymm5, %xmm6
	vpackssdw	%xmm6, %xmm5, %xmm5
	vpand	%xmm3, %xmm5, %xmm3
.LBB2_18:                               # %logical_op_done235
	vpsllw	$15, %xmm3, %xmm5
	vpmovmskb	%xmm5, %eax
	testl	$43690, %eax                    # imm = 0xAAAA
	je	.LBB2_14
# %bb.19:                               # %safe_if_run_true256
	vpmovzxwd	%xmm3, %ymm3            # ymm3 = xmm3[0],zero,xmm3[1],zero,xmm3[2],zero,xmm3[3],zero,xmm3[4],zero,xmm3[5],zero,xmm3[6],zero,xmm3[7],zero
	vpslld	$31, %ymm3, %ymm3
	vmaskmovps	%ymm0, %ymm3, (%r14,%rbx)
.LBB2_14:                               # %safe_if_after_true222
	vcmpnltps	%ymm2, %ymm4, %ymm0
	vandps	%ymm0, %ymm1, %ymm0
	vextractf128	$1, %ymm0, %xmm1
	vpackssdw	%xmm1, %xmm0, %xmm0
	vpmovmskb	%xmm0, %eax
	testw	%ax, %ax
	je	.LBB2_3
# %bb.15:                               # %safe_if_run_false276
	vpmovsxwd	%xmm0, %ymm0
	vbroadcastss	__real@4b189680(%rip), %ymm1 # ymm1 = [1.0E+7,1.0E+7,1.0E+7,1.0E+7,1.0E+7,1.0E+7,1.0E+7,1.0E+7]
	vmaskmovps	%ymm1, %ymm0, (%r14,%rbx)
.LBB2_3:                                # %foreach_reset
	vmovaps	80(%rsp), %xmm6                 # 16-byte Reload
	vmovaps	96(%rsp), %xmm7                 # 16-byte Reload
	vmovaps	112(%rsp), %xmm8                # 16-byte Reload
	vmovaps	128(%rsp), %xmm9                # 16-byte Reload
	vmovaps	144(%rsp), %xmm10               # 16-byte Reload
	vmovaps	160(%rsp), %xmm11               # 16-byte Reload
	vmovaps	176(%rsp), %xmm12               # 16-byte Reload
	vmovaps	192(%rsp), %xmm13               # 16-byte Reload
	vmovaps	208(%rsp), %xmm14               # 16-byte Reload
	vmovaps	224(%rsp), %xmm15               # 16-byte Reload
	addq	$240, %rsp
	popq	%rbx
	popq	%rbp
	popq	%rdi
	popq	%rsi
	popq	%r14
	vzeroupper
	retq
                                        # -- End function
	.section	.drectve,"yn"
	.ascii	" /FAILIFMISMATCH:\"_CRT_STDIO_ISO_WIDE_SPECIFIERS=0\""
	.globl	_fltused
