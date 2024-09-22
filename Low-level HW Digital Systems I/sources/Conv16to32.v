`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    20:51:04 01/05/2022 
// Design Name: 
// Module Name:    Conv16to32 
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
module Conv16to32(
     output reg [31:0] Out,
	  input wire [15:0] In,
	  input wire [5:0] Opcode
    );
	
	always @(In, Opcode)
		begin
			casez(Opcode)
			   //zero fill
				6'b11001?: Out = {16'h0000, In};          
				
				// zero fill to right with 16 bit left shift
				6'b111001: Out = {In, 16'h0000};          
				
				//sign extent - shift 2
				6'b1111??, 
				6'b00000?: Out = {{16{In[15]}}, In} << 2;
            
				//sign extent				
				default:   Out = {{16{In[15]}}, In};      
			endcase
		end

endmodule
