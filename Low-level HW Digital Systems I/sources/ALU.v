`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    20:58:09 01/03/2022 
// Design Name: 
// Module Name:    ALU 
// Project Name: 
// Target Devices: 
// Tool versions: 
// Description: 
//
// Dependencies: 
//
// Revision: 
// Revision 0.01 - File Created
// Additional Comments: 
//
//////////////////////////////////////////////////////////////////////////////////
module ALU(
	output reg  [31:0] Out,
	output reg Zero,
	input wire  [31:0] A, B,
	input wire [3:0] Op
    );
	
	always @(A, B, Op)
		begin
			case(Op)
				4'b0000: Out = A + B;
				4'b0001: Out = A - B;
				4'b0010: Out = A & B;
				4'b0011: Out = A | B;
				4'b0100: Out = ~A;
				4'b1000: Out = {A[31], A[31:1]};
				4'b1010: Out = {1'b0, A[31:1]};
				4'b1001: Out = {A[30:0], 1'b0};
				//4'b1001: Out = {1'b0, A[31:1]};
				//4'b1010: Out = {A[30:0], 1'b0};
				4'b1100: Out = {A[30:0], A[31]};
				4'b1101: Out = {A[0], A[31:1]};
				default: Out =32'hxxxxxxxx;
			endcase
			
			Zero = ~|Out;
		end

endmodule
