`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    17:49:53 01/04/2022 
// Design Name: 
// Module Name:    Register 
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
module Register(
	output reg [31:0] Dout,
	input wire [31:0] Data,
	input wire CLK, WE
    );
	 
	 reg [31:0] mem;
	 
	 always@(posedge CLK)
		begin
			if (WE == 1'b1)
				mem <= Data;
			else
				Dout <= mem;
		end

endmodule
